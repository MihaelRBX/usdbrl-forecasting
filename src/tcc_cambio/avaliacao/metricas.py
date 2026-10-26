"""Métricas de erro de previsão (metodologia, §3.6).

Todas recebem o realizado e o previsto **em nível de preço** (R$/US$) — ver
a decisão 2 em `avaliacao.protocolo`. A exceção é a acurácia direcional, que
olha só o sinal do retorno previsto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(real: np.ndarray, prev: np.ndarray) -> float:
    """Raiz do erro quadrático médio, na unidade da série (R$/US$)."""
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    return float(np.sqrt(np.mean((real - prev) ** 2)))


def mape(real: np.ndarray, prev: np.ndarray) -> float:
    """Erro percentual absoluto médio, em %.

    Só faz sentido em séries longe de zero — por isso é calculado sobre o preço
    (~R$ 2 a R$ 6) e nunca sobre o log-retorno.
    """
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    return float(np.mean(np.abs((real - prev) / real)) * 100.0)


def acuracia_direcional(y_real: np.ndarray, y_prev: np.ndarray) -> float:
    """Fração de origens em que o sinal do retorno previsto acerta o realizado.

    Previsões exatamente nulas não indicam direção e ficam de fora do cálculo;
    se todas forem nulas (caso do passeio aleatório), devolve NaN — "não se
    aplica", e não 0% ou 50%. Também é a métrica do Plano B da SNN.
    """
    y_real, y_prev = np.asarray(y_real, float), np.asarray(y_prev, float)
    com_direcao = y_prev != 0
    if not com_direcao.any():
        return float("nan")
    return float(np.mean(np.sign(y_prev[com_direcao]) == np.sign(y_real[com_direcao])))


def calcular(df: pd.DataFrame) -> pd.Series:
    """As três métricas para um grupo de previsões já juntado ao gabarito."""
    return pd.Series({
        "n": len(df),
        "rmse": rmse(df["preco_real"], df["preco_prev"]),
        "mape": mape(df["preco_real"], df["preco_prev"]),
        "acuracia_direcional": acuracia_direcional(df["y_real"], df["y_prev"]),
    })


def tabela_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de métricas por conjunto × horizonte × modelo.

    Para modelos com várias sementes, calcula as métricas **por semente** e
    reporta média e desvio-padrão entre elas (protocolo da metodologia). Para
    os determinísticos (uma semente só) o desvio fica NaN.
    """
    por_semente = (
        df.groupby(["conjunto", "h", "modelo", "semente"])
        .apply(calcular, include_groups=False)
        .reset_index()
        .astype({"n": int})
    )
    agregada = (
        por_semente.groupby(["conjunto", "h", "modelo"])
        .agg(
            sementes=("semente", "nunique"),
            n=("n", "first"),
            rmse=("rmse", "mean"),
            rmse_dp=("rmse", "std"),
            mape=("mape", "mean"),
            mape_dp=("mape", "std"),
            acuracia_direcional=("acuracia_direcional", "mean"),
        )
        .reset_index()
    )
    return agregada.sort_values(["conjunto", "h", "rmse"]).reset_index(drop=True)
