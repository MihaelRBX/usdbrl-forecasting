"""Tratamento da série bruta USD/BRL: de JSON cru a série tipada e validada.

Lê o artefato imutável de `data/raw/`, converte tipos, valida invariantes
estruturais e grava a série canônica em `data/processed/`.

Duas decisões de projeto que valem registro:

1. **Dias não-úteis não são valores faltantes.** A série do SGS só existe em
   dias úteis bancários brasileiros. Reindexar para dias corridos e preencher
   inventaria observações e infla artificialmente a autocorrelação. Aqui os
   intervalos são apenas *diagnosticados*, nunca preenchidos.

2. **Outliers são diagnosticados, não removidos.** Em série cambial os
   movimentos extremos são o próprio objeto de estudo — é deles que trata o
   componente GARCH. Removê-los apagaria a pandemia da amostra.

Uso:
    python -m tcc_cambio.dados.tratamento
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_cambio.config import ANO_FIM, ANO_INICIO, DIR_PROCESSED, DIR_RAW, SERIE_PADRAO

log = logging.getLogger(__name__)

# Formato das datas como o SGS as devolve. Explícito de propósito: sem isso,
# "02/01/2024" pode ser lido como 1º de fevereiro, e o erro é silencioso para
# todo dia menor ou igual a 12.
FORMATO_DATA_SGS = "%d/%m/%Y"

# Limites de sanidade para a cotação USD/BRL no período estudado.
COTACAO_MIN, COTACAO_MAX = 0.5, 20.0

# Retorno diário acima disso é sinalizado no diagnóstico (não removido).
LIMIAR_RETORNO_EXTREMO = 0.05


def verificar_integridade(caminho_json: Path) -> None:
    """Confere o SHA-256 do arquivo bruto contra o manifesto gravado na coleta.

    Garante que o insumo não foi alterado desde a coleta — a ponta que fecha a
    cadeia de proveniência.
    """
    caminho_manifesto = caminho_json.with_suffix(".manifesto.json")
    if not caminho_manifesto.exists():
        log.warning("manifesto ausente (%s): integridade não verificada",
                    caminho_manifesto.name)
        return

    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    atual = hashlib.sha256(caminho_json.read_bytes()).hexdigest()
    if atual != manifesto["sha256"]:
        raise ValueError(
            f"{caminho_json.name} não confere com o manifesto.\n"
            f"  esperado: {manifesto['sha256']}\n"
            f"  atual:    {atual}\n"
            "O arquivo bruto foi alterado — recolete em vez de editar."
        )
    log.info("integridade conferida: %s (%d registros no manifesto)",
             caminho_json.name, manifesto["n_registros"])


def carregar_bruto(caminho_json: Path) -> pd.DataFrame:
    """Lê o JSON bruto e devolve um DataFrame tipado e ordenado."""
    registros = json.loads(caminho_json.read_text(encoding="utf-8"))
    df = pd.DataFrame(registros)

    # Conversões explícitas. O `format=` não é opcional: ver FORMATO_DATA_SGS.
    df["data"] = pd.to_datetime(df["data"], format=FORMATO_DATA_SGS)
    df["cotacao"] = pd.to_numeric(df["valor"], errors="raise")

    df = (df[["data", "cotacao"]]
          .sort_values("data")
          .reset_index(drop=True))

    log.info("carregados %d registros de %s a %s",
             len(df), df["data"].iloc[0].date(), df["data"].iloc[-1].date())
    return df


def validar(df: pd.DataFrame) -> None:
    """Verifica as invariantes estruturais. Levanta na primeira violação.

    São condições que, se quebradas, invalidam qualquer modelagem posterior —
    melhor falhar aqui do que descobrir depois de treinar.
    """
    if df.empty:
        raise ValueError("série vazia.")

    n_dup = int(df["data"].duplicated().sum())
    if n_dup:
        exemplos = df.loc[df["data"].duplicated(keep=False), "data"].head(5).tolist()
        raise ValueError(f"{n_dup} data(s) duplicada(s). Exemplos: {exemplos}")

    if not df["data"].is_monotonic_increasing:
        raise ValueError("série fora de ordem cronológica.")

    n_nulos = int(df["cotacao"].isna().sum())
    if n_nulos:
        raise ValueError(f"{n_nulos} cotação(ões) nula(s).")

    fora = df[(df["cotacao"] <= COTACAO_MIN) | (df["cotacao"] >= COTACAO_MAX)]
    if not fora.empty:
        raise ValueError(
            f"{len(fora)} cotação(ões) fora da faixa plausível "
            f"({COTACAO_MIN}–{COTACAO_MAX}): {fora.head(5).to_dict('records')}"
        )

    log.info("validação estrutural: OK (%d observações)", len(df))


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta o log-retorno diário.

    log(P_t) - log(P_{t-1}). É o alvo estacionário natural: é o que o
    ARMA-GARCH modela e o que a codificação por eventos da SNN representa.
    A primeira observação fica NaN por construção — não há retorno sem
    referência anterior. Manter a linha preserva a série de preços íntegra;
    quem usa o retorno descarta a primeira.
    """
    df = df.copy()
    df["log_retorno"] = np.log(df["cotacao"]).diff()
    return df


def diagnosticar(df: pd.DataFrame) -> None:
    """Relata características da série. NÃO altera nada.

    Separado de `validar` de propósito: aqui nada é erro, é caracterização —
    insumo para a análise exploratória do TCC.
    """
    ret = df["log_retorno"].dropna()

    log.info("--- diagnóstico ---")
    log.info("período            : %s a %s",
             df["data"].iloc[0].date(), df["data"].iloc[-1].date())
    log.info("observações        : %d", len(df))
    log.info("cotação            : min %.4f | méd %.4f | max %.4f",
             df["cotacao"].min(), df["cotacao"].mean(), df["cotacao"].max())
    log.info("log-retorno diário : méd %+.6f | desvio %.6f",
             ret.mean(), ret.std())
    log.info("volatilidade anual : %.2f%%", ret.std() * np.sqrt(252) * 100)

    # Intervalos entre observações. Saltos de 3 a 5 dias são normais (fins de
    # semana e feriados). Acima disso vale conferir no calendário.
    dias = df["data"].diff().dt.days.dropna()
    grandes = dias[dias > 5]
    log.info("maior intervalo    : %d dias corridos", int(dias.max()))
    log.info("intervalos > 5 dias: %d", len(grandes))

    # Movimentos extremos: reportados para a análise por regime, não removidos.
    extremos = df.loc[df["log_retorno"].abs() > LIMIAR_RETORNO_EXTREMO]
    log.info("retornos |.| > %.0f%%  : %d", LIMIAR_RETORNO_EXTREMO * 100, len(extremos))
    for _, linha in extremos.nlargest(5, columns="log_retorno", keep="all").head(5).iterrows():
        log.info("    %s  %+.2f%%  (R$ %.4f)",
                 linha["data"].date(), linha["log_retorno"] * 100, linha["cotacao"])


def salvar_processado(df: pd.DataFrame, caminho: Path) -> Path:
    """Grava a série canônica em CSV com datas em ISO-8601."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, date_format="%Y-%m-%d")
    log.info("gravado %s (%d linhas)", caminho.name, len(df))
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trata a série bruta do SGS e grava a série canônica."
    )
    parser.add_argument("--entrada", type=Path,
                        default=DIR_RAW / f"sgs_{SERIE_PADRAO}_{ANO_INICIO}_{ANO_FIM}.json",
                        help="JSON bruto produzido pela coleta")
    parser.add_argument("--saida", type=Path,
                        default=DIR_PROCESSED / "usdbrl_diario.csv",
                        help="CSV da série tratada")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.entrada.exists():
        parser.error(f"arquivo não encontrado: {args.entrada}\n"
                     "Rode antes: python -m tcc_cambio.dados.coleta_sgs")

    verificar_integridade(args.entrada)
    df = carregar_bruto(args.entrada)
    validar(df)
    df = enriquecer(df)
    diagnosticar(df)
    caminho = salvar_processado(df, args.saida)

    print(f"\n{len(df)} observações de {df['data'].iloc[0].date()} "
          f"a {df['data'].iloc[-1].date()}")
    print(f"-> {caminho}")


if __name__ == "__main__":
    main()
