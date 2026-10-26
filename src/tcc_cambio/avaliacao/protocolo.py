"""Protocolo de previsão: o que cada modelo prevê, a partir de quando, e em que formato.

Este módulo é o **contrato comum** entre os quatro modelos e a avaliação. Se
cada modelo gerasse previsões do seu jeito, RMSE/MAPE e Diebold-Mariano
estariam comparando coisas diferentes. As decisões abaixo valem para todos:

1. **Alvo: log-retorno acumulado em h dias úteis.**
       y(t, h) = log(P[t+h]) - log(P[t]) = r[t+1] + ... + r[t+h]
   Previsão **direta** de cada horizonte h ∈ HORIZONTES. O log-retorno é
   estacionário (ver `analise.diagnosticos`); o nível de preço não é.

2. **Métricas sobre o preço reconstruído.**
       P̂[t+h] = P[t] · exp(ŷ(t, h))
   RMSE e MAPE são calculados em R$/US$, não no retorno: o retorno oscila em
   torno de zero e o MAPE sobre ele explode (divisão por ~0). Reconstruir o
   preço também deixa o erro interpretável ("erra em média R$ 0,03").

3. **Origem móvel (rolling origin).** Cada dia t do conjunto avaliado é uma
   origem de previsão. Na origem t o modelo só pode usar dados até t
   (inclusive) — inclusive dados do próprio conjunto anteriores a t, que já
   seriam conhecidos naquele dia. Parâmetros (pesos das redes, coeficientes do
   ARMA-GARCH, normalização, limiares de codificação da SNN) são estimados
   **só no treino**; se algum modelo reestimar parâmetros ao longo do teste,
   isso precisa ser declarado no texto.

4. **O realizado não atravessa a fronteira do conjunto.** A origem t só entra
   se t+h ainda pertence ao mesmo conjunto. Sem isso, a avaliação na
   validação usaria cotações do teste como gabarito.

5. **Formato comum de saída.** Cada modelo grava um CSV em
   `data/resultados/previsoes_<modelo>.csv` com as colunas de COLUNAS_PREVISAO.
   `avaliacao.avaliar` junta tudo com o gabarito e calcula as métricas.

Uso típico, dentro de um módulo de modelo:

    serie = carregar_serie_dividida()
    gab = gabarito(serie, "teste", h=5)
    y_prev = ...                       # um valor por linha de `gab`
    prev = montar_previsoes(gab, y_prev, modelo="gru", semente=42)
    salvar_previsoes(prev, "gru")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_cambio.config import DIR_PROCESSED, DIR_RESULTADOS, HORIZONTES

log = logging.getLogger(__name__)

# Conjuntos em que se gera previsão. O treino não entra: é onde os parâmetros
# são estimados, então o erro ali não mede capacidade de previsão.
CONJUNTOS_AVALIADOS = ("validacao", "teste")

# Colunas que todo arquivo de previsões precisa ter. `semente` é -1 para
# modelos determinísticos (passeio aleatório, ARMA-GARCH).
COLUNAS_PREVISAO = ("modelo", "semente", "conjunto", "h", "data_origem", "y_prev")

SEMENTE_DETERMINISTICO = -1


def carregar_serie_dividida(caminho: Path | None = None) -> pd.DataFrame:
    """Lê a série inteira já rotulada (saída de `dados.split`), em ordem cronológica.

    Diferente de `analise.diagnosticos.carregar_serie`, aqui **não** se filtra
    por conjunto: o modelo precisa do histórico completo para montar a entrada
    de cada origem; quem restringe o que é avaliado é `gabarito`.
    """
    caminho = caminho or DIR_PROCESSED / "usdbrl_split.csv"
    df = pd.read_csv(caminho, parse_dates=["data"])
    faltando = {"data", "cotacao", "log_retorno", "conjunto"} - set(df.columns)
    if faltando:
        raise ValueError(
            f"colunas ausentes em {caminho.name}: {faltando}. "
            "Rode antes: python -m tcc_cambio.dados.split"
        )
    if not df["data"].is_monotonic_increasing:
        raise ValueError(f"{caminho.name} fora de ordem cronológica")
    return df.reset_index(drop=True)


def gabarito(serie: pd.DataFrame, conjunto: str, h: int) -> pd.DataFrame:
    """Origens de previsão de um conjunto para o horizonte h, com o valor realizado.

    Uma linha por origem t válida (t e t+h no mesmo `conjunto`). Colunas:
      - `idx_origem`: posição de t em `serie` (o modelo usa dados até aqui);
      - `data_origem`, `preco_origem`: data e cotação P[t];
      - `data_alvo`, `preco_real`: data e cotação P[t+h];
      - `y_real`: log(P[t+h] / P[t]), o alvo.
    """
    if conjunto not in CONJUNTOS_AVALIADOS:
        raise ValueError(f"conjunto deve ser um de {CONJUNTOS_AVALIADOS}, não '{conjunto}'")
    if h < 1:
        raise ValueError(f"horizonte deve ser >= 1, não {h}")

    preco = serie["cotacao"].to_numpy()
    rotulo = serie["conjunto"].to_numpy()
    n = len(serie)

    idx = np.flatnonzero(rotulo == conjunto)
    idx = idx[idx + h < n]
    idx = idx[rotulo[idx + h] == conjunto]

    gab = pd.DataFrame({
        "conjunto": conjunto,
        "h": h,
        "idx_origem": idx,
        "data_origem": serie["data"].to_numpy()[idx],
        "preco_origem": preco[idx],
        "data_alvo": serie["data"].to_numpy()[idx + h],
        "preco_real": preco[idx + h],
    })
    gab["y_real"] = np.log(gab["preco_real"] / gab["preco_origem"])
    return gab


def gabaritos(serie: pd.DataFrame) -> pd.DataFrame:
    """Todos os gabaritos (conjuntos avaliados × horizontes) num DataFrame só."""
    return pd.concat(
        [gabarito(serie, c, h) for c in CONJUNTOS_AVALIADOS for h in HORIZONTES],
        ignore_index=True,
    )


def montar_previsoes(
    gab: pd.DataFrame,
    y_prev: np.ndarray | pd.Series,
    modelo: str,
    semente: int = SEMENTE_DETERMINISTICO,
) -> pd.DataFrame:
    """Empacota as previsões de um modelo no formato comum.

    `y_prev` é o log-retorno acumulado previsto, alinhado linha a linha com
    `gab` (mesmo comprimento, mesma ordem).
    """
    y_prev = np.asarray(y_prev, dtype=float)
    if y_prev.shape != (len(gab),):
        raise ValueError(f"y_prev tem forma {y_prev.shape}; esperado ({len(gab)},)")
    if not np.isfinite(y_prev).all():
        raise ValueError(f"{modelo}: previsões com NaN/inf")

    prev = gab[["conjunto", "h", "data_origem"]].copy()
    prev.insert(0, "modelo", modelo)
    prev.insert(1, "semente", semente)
    prev["y_prev"] = y_prev
    return prev[list(COLUNAS_PREVISAO)]


def salvar_previsoes(prev: pd.DataFrame, modelo: str, diretorio: Path | None = None) -> Path:
    """Grava as previsões de um modelo (todas as sementes, conjuntos e horizontes)."""
    faltando = set(COLUNAS_PREVISAO) - set(prev.columns)
    if faltando:
        raise ValueError(f"colunas ausentes nas previsões: {faltando}")
    duplicadas = prev.duplicated(["semente", "conjunto", "h", "data_origem"])
    if duplicadas.any():
        raise ValueError(f"{int(duplicadas.sum())} previsões duplicadas para a mesma origem")

    caminho = (diretorio or DIR_RESULTADOS) / f"previsoes_{modelo}.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    prev[list(COLUNAS_PREVISAO)].to_csv(caminho, index=False, date_format="%Y-%m-%d")
    log.info("gravado %s (%d previsões)", caminho.name, len(prev))
    return caminho


def carregar_previsoes(diretorio: Path | None = None) -> pd.DataFrame:
    """Lê todos os `previsoes_*.csv` do diretório num DataFrame só."""
    diretorio = diretorio or DIR_RESULTADOS
    arquivos = sorted(diretorio.glob("previsoes_*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"nenhum previsoes_*.csv em {diretorio}. "
            "Rode antes algum modelo, ex.: python -m tcc_cambio.modelos.passeio_aleatorio"
        )
    return pd.concat(
        [pd.read_csv(a, parse_dates=["data_origem"]) for a in arquivos],
        ignore_index=True,
    )


def juntar_com_gabarito(prev: pd.DataFrame, gab: pd.DataFrame) -> pd.DataFrame:
    """Casa previsões com o realizado e reconstrói o preço previsto.

    Falha se alguma previsão não tiver gabarito correspondente — sinal de que o
    modelo previu numa origem fora do protocolo (ex.: atravessando a fronteira).
    """
    chaves = ["conjunto", "h", "data_origem"]
    df = prev.merge(gab, on=chaves, how="left", validate="many_to_one")
    orfas = df["y_real"].isna()
    if orfas.any():
        exemplo = df.loc[orfas, chaves].iloc[0].to_dict()
        raise ValueError(
            f"{int(orfas.sum())} previsões sem gabarito (fora do protocolo). Ex.: {exemplo}"
        )
    df["preco_prev"] = df["preco_origem"] * np.exp(df["y_prev"])
    return df
