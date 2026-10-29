"""Diagnósticos de série temporal da USD/BRL: o fundamento do ARMA-GARCH.

Segunda peça da análise exploratória (metodologia, §3.3). Enquanto os
histogramas *mostram* a distribuição, aqui a série é *testada* — é esta etapa
que **justifica formalmente** a escolha do baseline econométrico e fixa as
ordens dos modelos. Três perguntas, três blocos de teste:

1. **A série é estacionária?** A cotação (nível de preço) não é: tem tendência
   e raiz unitária. O log-retorno é. ADF e KPSS respondem isso com hipóteses
   nulas *opostas* — usá-los em conjunto é confirmação cruzada, não redundância.

2. **Há estrutura autorregressiva a modelar?** ACF e PACF do log-retorno
   sugerem as ordens (q, p) do ARMA; Ljung-Box testa se a autocorrelação
   remanescente é conjuntamente significativa.

3. **A volatilidade é condicionalmente heterocedástica?** Ljung-Box no
   *quadrado* dos retornos e o teste ARCH-LM de Engle detectam
   agrupamento de volatilidade — **a justificativa empírica direta do
   componente GARCH**. Jarque-Bera fecha o argumento das caudas pesadas já
   visto no histograma.

Nota metodológica — vazamento: os testes que informam decisões de modelagem
rodam sobre o **conjunto de treino**, não sobre a série inteira. Caracterizar
validação/teste aqui seria espiar o futuro. Por isso `carregar_serie` filtra
por `conjunto` e lê o arquivo já dividido (`usdbrl_split.csv`).

Dependências: grupo `econometria` (statsmodels) e `dev` (matplotlib).

Uso:
    python -m tcc_cambio.analise.diagnosticos
    python -m tcc_cambio.analise.diagnosticos --conjunto treino
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import warnings

import matplotlib

matplotlib.use("Agg")  # backend sem display: roda em terminal e em CI.

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from tcc_cambio.config import DIR_FIGURAS, DIR_PROCESSED, DIR_TABELAS

log = logging.getLogger(__name__)

# Nível de significância padrão dos testes de hipótese. 5% é a convenção do
# texto do TCC; toda decisão "rejeita/não rejeita" abaixo usa este limiar.
ALFA = 0.05

# Defasagens padrão para Ljung-Box e ARCH-LM. ~2 semanas de pregão (10 dias
# úteis) cobre o horizonte em que se espera ver agrupamento de volatilidade
# diária sem diluir o teste em ruído de longo prazo.
DEFASAGENS_PADRAO = 10

# Defasagens exibidas nos correlogramas ACF/PACF. ~1 mês de pregão.
MAX_LAG_CORRELOGRAMA = 20


def carregar_serie(caminho: Path, conjunto: str | None = "treino") -> pd.DataFrame:
    """Lê a série dividida (saída de `dados.split`) e filtra por conjunto.

    Implementar:
      - ler o CSV com `parse_dates=["data"]`;
      - validar que as colunas {"cotacao", "log_retorno", "conjunto"} existem —
        senão, erro orientando a rodar `python -m tcc_cambio.dados.split`;
      - se `conjunto` não for None, filtrar `df[df.conjunto == conjunto]` e
        exigir que o rótulo exista (erro claro se digitado errado);
      - logar quantas observações e o intervalo de datas do recorte retornado.

    Devolve o DataFrame filtrado, com o índice reiniciado.
    """
    df = pd.read_csv(caminho, parse_dates=["data"])

    faltando = {"cotacao", "log_retorno", "conjunto"} - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes em {caminho.name}: {faltando}. "
            "Rode antes: python -m tcc_cambio.dados.split"
        )

    rotulos = (df["conjunto"].unique())

    if conjunto is None:
        recortado = df.reset_index(drop=True)
    else:
        if conjunto not in rotulos:
            raise ValueError(
                f"Parâmetro 'conjunto' passado invalidamente como '{conjunto}'.\n"
                f"O Parâmetro 'conjunto' deve ser um dos seguintes: {rotulos}"
            )
        
        recortado= df[df.conjunto == conjunto].reset_index(drop=True)

    log.info(
            f"Cotações retornadas no recorte: {len(recortado)}.\n"
            f"Período do recorte: {recortado["data"].min().date()} até {recortado["data"].max().date()}."
        )
    
    return recortado


def teste_adf(serie: pd.Series) -> dict:
    """Teste Augmented Dickey-Fuller de raiz unitária.

    H0: a série tem raiz unitária (**não-estacionária**).
    H1: a série é estacionária.
    Rejeitar H0 (p < ALFA) é evidência de estacionariedade.

    Implementar com `statsmodels.tsa.stattools.adfuller`. A `serie` deve vir
    sem NaN (o log-retorno tem o primeiro valor nulo por construção — usar
    `.dropna()` antes de chamar, ou aqui dentro).

    Devolver um dict com pelo menos: `estatistica`, `p_valor`, `defasagens`
    (lags usados), `n_obs`, `valores_criticos` (dict 1%/5%/10%) e
    `estacionaria` (bool = p_valor < ALFA). Esse dict alimenta a tabela de
    `resumir`.
    """
    serie = serie.dropna()

    estatistica, p_valor, defasagens, n_obs, valores_criticos, _ = adfuller(
        serie, result_object=False
    )

    return {
        "estatistica": estatistica,
        "p_valor": p_valor,
        "defasagens": defasagens,
        "n_obs": n_obs,
        "valores_criticos": valores_criticos,
        "estacionaria": p_valor < ALFA,
    }


def teste_kpss(serie: pd.Series) -> dict:
    """Teste KPSS de estacionariedade.

    H0: a série é **estacionária** (em torno de nível ou tendência).
    H1: a série tem raiz unitária.
    Note que as hipóteses são o *oposto* do ADF: aqui rejeitar H0 (p < ALFA) é
    evidência de NÃO-estacionariedade. Usar `regression="c"` (estacionária em
    torno de constante) para o log-retorno.

    Implementar com `statsmodels.tsa.stattools.kpss` (silenciar/atender o
    InterpolationWarning quando o p-valor satura nos extremos da tabela).

    Devolver dict análogo ao de `teste_adf`, com `estacionaria` = p_valor > ALFA
    (cuidado: a lógica é invertida em relação ao ADF). ADF e KPSS concordando é
    o que dá segurança para a afirmação de estacionariedade no texto.
    """
    serie = serie.dropna()

    # O p-valor do KPSS vem de interpolação numa tabela que só vai de 0,01 a
    # 0,10. Fora dela o statsmodels devolve o próprio limite e avisa; o aviso
    # vira a chave `p_no_limite`, que `resumir` traduz em "< 0,01" / "> 0,10".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        estatistica, p_valor, defasagens, valores_criticos = kpss(
            serie, regression="c", nlags="auto", result_object=False
        )

    return {
        "estatistica": estatistica,
        "p_valor": p_valor,
        "defasagens": defasagens,
        "n_obs": len(serie),
        "valores_criticos": valores_criticos,
        "p_no_limite": p_valor in (0.01, 0.10),
        "estacionaria": p_valor > ALFA,
    }


def teste_ljung_box(serie: pd.Series, defasagens: int = DEFASAGENS_PADRAO) -> pd.DataFrame:
    """Teste de Ljung-Box de autocorrelação conjunta.

    H0: as autocorrelações até `defasagens` são todas nulas (sem estrutura
    serial). Rejeitar (p < ALFA) indica que há dependência temporal a modelar.

    Aplicar em dois alvos, chamando esta função duas vezes:
      - na **própria série** de log-retornos → detecta memória linear (ARMA);
      - no **quadrado** dos log-retornos → detecta agrupamento de volatilidade
        (efeito ARCH). É este segundo uso que sustenta o GARCH.

    Implementar com `statsmodels.stats.diagnostic.acorr_ljungbox(serie,
    lags=defasagens, return_df=True)`. Devolver o DataFrame resultante (uma
    linha por defasagem, colunas de estatística e p-valor).
    """
    return acorr_ljungbox(serie.dropna(), lags=defasagens, return_df=True)


def teste_arch_lm(serie: pd.Series, defasagens: int = DEFASAGENS_PADRAO) -> dict:
    """Teste ARCH-LM de Engle para heterocedasticidade condicional.

    H0: não há efeito ARCH (variância condicional constante — homocedástica).
    Rejeitar (p < ALFA) é a **evidência direta e formal de que o GARCH é
    necessário** — complementa o Ljung-Box nos retornos ao quadrado.

    Implementar com `statsmodels.stats.diagnostic.het_arch(serie,
    nlags=defasagens)`, que devolve (LM, p_LM, F, p_F). Aplicar sobre o
    log-retorno (já centrado, ou centrar subtraindo a média).

    Devolver dict com `estatistica_lm`, `p_valor_lm`, `estatistica_f`,
    `p_valor_f` e `tem_efeito_arch` (bool = p_valor_lm < ALFA).
    """
    serie = serie.dropna()

    estatistica_lm, p_valor_lm, estatistica_f, p_valor_f = het_arch(
        serie - serie.mean(), nlags=defasagens, result_object=False
    )

    return {
        "estatistica_lm": estatistica_lm,
        "p_valor_lm": p_valor_lm,
        "estatistica_f": estatistica_f,
        "p_valor_f": p_valor_f,
        "defasagens": defasagens,
        "tem_efeito_arch": p_valor_lm < ALFA,
    }


def teste_jarque_bera(serie: pd.Series) -> dict:
    """Teste de normalidade de Jarque-Bera (via assimetria e curtose).

    H0: os dados vêm de uma normal. Rejeitar (p < ALFA) confirma o afastamento
    da normalidade — o mesmo leptocurtismo já anotado no histograma, agora com
    um teste formal por trás. Reforça a escolha de distribuição de erros de
    cauda pesada (ex.: t de Student) no GARCH.

    Implementar com `statsmodels.stats.stattools.jarque_bera`, que devolve
    (JB, p, assimetria, curtose). Devolver dict com esses campos e `normal`
    (bool = p_valor > ALFA).
    """
    serie = serie.dropna()

    # `curtose` aqui é a curtose comum (normal = 3), não a "em excesso" do
    # histograma; as duas ficam no dict para não haver confusão no texto.
    estatistica, p_valor, assimetria, curtose = jarque_bera(serie)

    return {
        "estatistica": float(estatistica),
        "p_valor": float(p_valor),
        "assimetria": float(assimetria),
        "curtose": float(curtose),
        "curtose_excesso": float(curtose) - 3.0,
        "normal": p_valor > ALFA,
    }


def plot_acf_pacf(serie: pd.Series, max_lag: int, caminho: Path) -> Path:
    """Correlogramas ACF e PACF do log-retorno, lado a lado.

    Leitura que orienta as ordens do ARMA(p, q):
      - a **ACF** que corta após a defasagem q sugere um MA(q);
      - a **PACF** que corta após a defasagem p sugere um AR(p).
    As bandas de confiança (±1.96/√n) marcam o que é ruído.

    Implementar:
      - `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))`;
      - `statsmodels.graphics.tsaplots.plot_acf(serie, lags=max_lag, ax=ax1)` e
        `plot_pacf(serie, lags=max_lag, ax=ax2, method="ywm")`;
      - títulos/rótulos em português, `tight_layout`, criar o diretório-pai,
        salvar com `dpi=150` e `plt.close(fig)`.

    Devolver o `caminho` gravado (padrão do módulo `analise`).
    """
    serie = serie.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # zero=False: a defasagem 0 (correlação 1 por definição) só achataria o
    # gráfico; auto_ylims ajusta o eixo às correlações reais, que são pequenas.
    plot_acf(serie, lags=max_lag, ax=ax1, zero=False, auto_ylims=True)
    plot_pacf(serie, lags=max_lag, ax=ax2, method="ywm", zero=False, auto_ylims=True)

    ax1.set_title("ACF — autocorrelação")
    ax2.set_title("PACF — autocorrelação parcial")
    for ax in (ax1, ax2):
        ax.set_xlabel("Defasagem (dias úteis)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Correlação")
    fig.suptitle(f"Correlogramas do {serie.name or 'log-retorno'} diário USD/BRL "
                 f"(n = {serie.size})")
    fig.tight_layout()

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    log.info("gravado %s", caminho.name)
    return caminho


# Por teste: hipótese nula e a leitura de "rejeita" / "não rejeita" H0.
_LEITURAS = {
    "ADF": ("raiz unitária", "estacionária", "não estacionária"),
    "KPSS": ("estacionária", "não estacionária", "estacionária"),
    "Ljung-Box": ("sem autocorrelação até a defasagem k",
                  "há dependência serial", "sem dependência serial"),
    "ARCH-LM": ("sem efeito ARCH", "há efeito ARCH", "sem efeito ARCH"),
    "Jarque-Bera": ("normalidade", "não normal", "normal"),
}


def _extrair(teste: str, resultado: dict | pd.DataFrame) -> tuple:
    """(estatística, p-valor, H0, leitura se rejeita, leitura se não rejeita)."""
    if teste.startswith("Ljung-Box"):
        # Uma linha por defasagem; a tabela usa a última (teste conjunto até k).
        ultima = resultado.iloc[-1]
        estatistica, p_valor = ultima["lb_stat"], ultima["lb_pvalue"]
    elif teste == "ARCH-LM":
        estatistica, p_valor = resultado["estatistica_lm"], resultado["p_valor_lm"]
    else:
        estatistica, p_valor = resultado["estatistica"], resultado["p_valor"]
    h0, sim, nao = _LEITURAS[teste.split(" ")[0]]
    return float(estatistica), float(p_valor), h0, sim, nao


def _formatar_p(p_valor: float, resultado) -> str:
    """p-valor para o texto: respeita os limites da tabela do KPSS e p ~ 0."""
    if isinstance(resultado, dict) and resultado.get("p_no_limite"):
        return "< 0,01" if p_valor <= 0.01 else "> 0,10"
    if p_valor < 1e-4:
        return "< 0,0001"
    return f"{p_valor:.4f}".replace(".", ",")


def resumir(
    diagnosticos: dict,
    caminho: Path | None = None,
) -> pd.DataFrame:
    """Consolida os testes numa tabela única e a loga (e opcionalmente grava).

    Recebe os dicts/DataFrames devolvidos pelos testes acima (agrupados no
    `main`) e monta uma tabela legível: uma linha por teste, colunas de
    estatística, p-valor e decisão ("rejeita H0" / "não rejeita"). É material
    direto para a tabela de diagnósticos da metodologia do TCC.

    Se `caminho` for dado, gravar em CSV (padrão do projeto: datas ISO, sem
    índice). Logar a tabela com `to_string(index=False)`. Devolver o DataFrame.
    """
    linhas = []
    for (teste, serie), resultado in diagnosticos.items():
        estatistica, p_valor, h0, rejeita_sim, rejeita_nao = _extrair(teste, resultado)
        if teste.startswith("Ljung-Box") and serie.endswith("²"):
            # No quadrado do retorno, dependência serial = volatilidade em blocos.
            rejeita_sim, rejeita_nao = "agrupamento de volatilidade", "sem agrupamento"
        rejeita = p_valor < ALFA
        linhas.append({
            "teste": teste,
            "serie": serie,
            "h0": h0,
            "estatistica": estatistica,
            "p_valor": p_valor,
            "p_texto": _formatar_p(p_valor, resultado),
            "decisao": "rejeita H0" if rejeita else "não rejeita H0",
            "conclusao": rejeita_sim if rejeita else rejeita_nao,
        })

    tabela = pd.DataFrame(linhas)
    log.info("diagnósticos (α = %.2f):\n%s\n", ALFA,
             tabela.drop(columns="p_valor").to_string(
                 index=False, float_format=lambda x: f"{x:.3f}"))

    if caminho is not None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(caminho, index=False)
        log.info("gravado %s", caminho.name)

    return tabela


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnósticos de série temporal da USD/BRL "
                    "(estacionariedade, autocorrelação e efeitos ARCH)."
    )
    parser.add_argument("--entrada", type=Path,
                        default=DIR_PROCESSED / "usdbrl_split.csv",
                        help="CSV da série dividida (saída de dados.split)")
    parser.add_argument("--conjunto", default="treino",
                        help="conjunto a diagnosticar; 'todos' usa a série inteira")
    parser.add_argument("--saida", type=Path, default=DIR_FIGURAS,
                        help="diretório para a figura ACF/PACF")
    parser.add_argument("--tabelas", type=Path, default=DIR_TABELAS,
                        help="diretório para a tabela de diagnósticos")
    parser.add_argument("--defasagens", type=int, default=DEFASAGENS_PADRAO,
                        help="defasagens de Ljung-Box e ARCH-LM")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.entrada.exists():
        parser.error(f"arquivo não encontrado: {args.entrada}\n"
                     "Rode antes: python -m tcc_cambio.dados.split")

    conjunto = None if args.conjunto == "todos" else args.conjunto
    df = carregar_serie(args.entrada, conjunto=conjunto)
    cotacao = df["cotacao"]
    retorno = df["log_retorno"].dropna().rename("log-retorno")
    k = args.defasagens

    diagnosticos = {
        ("ADF", "cotação"): teste_adf(cotacao),
        ("ADF", "log-retorno"): teste_adf(retorno),
        ("KPSS", "cotação"): teste_kpss(cotacao),
        ("KPSS", "log-retorno"): teste_kpss(retorno),
        (f"Ljung-Box ({k})", "log-retorno"): teste_ljung_box(retorno, k),
        (f"Ljung-Box ({k})", "log-retorno²"): teste_ljung_box(retorno**2, k),
        ("ARCH-LM", "log-retorno"): teste_arch_lm(retorno, k),
        ("Jarque-Bera", "log-retorno"): teste_jarque_bera(retorno),
    }

    sufixo = "" if args.conjunto == "treino" else f"_{args.conjunto}"
    figura = plot_acf_pacf(retorno, MAX_LAG_CORRELOGRAMA,
                           args.saida / f"acf_pacf_log_retorno{sufixo}.png")
    tabela = args.tabelas / f"diagnosticos{sufixo}.csv"
    resumir(diagnosticos, tabela)

    print(f"\n-> {figura}\n-> {tabela}")


if __name__ == "__main__":
    main()
