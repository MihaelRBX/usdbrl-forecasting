"""Teste de Diebold-Mariano com correções para horizonte e múltiplas comparações.

RMSE menor não basta: o DM diz se a diferença de erro entre dois modelos é
estatisticamente significativa (Diebold e Mariano, 1995).

    d[t] = L(e1[t]) - L(e2[t])          diferencial de perda, L = erro²
    DM   = d̄ / sqrt(V̂(d̄))              H0: E[d] = 0 (precisão igual)

Três detalhes que pesam neste TCC:

1. **Previsões sobrepostas (h > 1).** Com uma origem por dia e alvo de h dias,
   origens vizinhas compartilham h-1 retornos — o erro é MA(h-1) mesmo para
   um modelo ótimo. A variância de d̄ inclui as autocovariâncias até h-1.

2. **Amostra finita.** Usa-se a correção de Harvey, Leybourne e Newbold (1997)
   e a distribuição t de Student com T-1 graus de liberdade.

3. **Múltiplas comparações.** 4 modelos (+ passeio aleatório) geram vários
   pares em cada horizonte. Os p-valores são ajustados por **Holm-Bonferroni**
   (Holm, 1979) dentro de cada conjunto avaliado e horizonte — a família é o
   conjunto de pares de um mesmo horizonte (base, 06-metricas-e-avaliacao).

Para modelos com várias sementes, o teste usa a **previsão média entre
sementes** (metodologia, "Múltiplas sementes").
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ALFA = 0.05

# Colunas da tabela de `comparar_todos`. Fixas para que o CSV tenha cabeçalho
# mesmo quando ainda não há pares a comparar (um modelo só).
COLUNAS = (
    "conjunto", "h", "modelo_1", "modelo_2", "estatistica", "p_valor", "t",
    "p_holm", "significativo", "melhor",
)


def _autocovariancia(d: np.ndarray, k: int) -> float:
    """Autocovariância amostral de defasagem k (divisor T, como em DM 1995)."""
    t = d.size
    dc = d - d.mean()
    return float(np.dot(dc[k:], dc[: t - k]) / t)


def teste_dm(e1: np.ndarray, e2: np.ndarray, h: int) -> dict:
    """Diebold-Mariano (com correção HLN) para erros de dois modelos.

    `e1` e `e2` são os erros de previsão (real - previsto) nas mesmas origens.
    Estatística negativa: o modelo 1 erra menos que o modelo 2.

    Se a variância de longo prazo com pesos retangulares sair negativa (pode
    acontecer em amostras pequenas), recai nos pesos de Bartlett (Newey-West),
    que garantem positividade.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if e1.shape != e2.shape:
        raise ValueError("e1 e e2 precisam ter o mesmo tamanho (mesmas origens)")

    d = e1**2 - e2**2
    t = d.size
    d_media = float(d.mean())

    gamas = [_autocovariancia(d, k) for k in range(h)]
    variancia_lp = gamas[0] + 2.0 * sum(gamas[1:])
    if variancia_lp <= 0:
        variancia_lp = gamas[0] + 2.0 * sum((1 - k / h) * g for k, g in enumerate(gamas[1:], 1))

    if variancia_lp <= 0:  # previsões idênticas: d ≡ 0
        return {"estatistica": float("nan"), "p_valor": float("nan"), "t": t}

    dm = d_media / np.sqrt(variancia_lp / t)
    correcao_hln = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    estatistica = float(dm * correcao_hln)
    p_valor = float(2.0 * stats.t.sf(abs(estatistica), df=t - 1))
    return {"estatistica": estatistica, "p_valor": p_valor, "t": t}


def holm(p_valores: pd.Series) -> pd.Series:
    """Ajuste de Holm-Bonferroni (step-down), preservando o índice de entrada."""
    p = p_valores.dropna()
    m = len(p)
    ordem = p.sort_values()
    ajustado = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ordem.to_numpy()))
    return pd.Series(ajustado, index=ordem.index).reindex(p_valores.index)


def comparar_todos(df: pd.DataFrame) -> pd.DataFrame:
    """DM para todos os pares de modelos, por conjunto e horizonte.

    `df` é a saída de `protocolo.juntar_com_gabarito`. O erro é medido no
    preço reconstruído, coerente com RMSE/MAPE. Devolve uma linha por
    (conjunto, h, par) com estatística, p-valor bruto, p-valor de Holm e a
    decisão ao nível ALFA.
    """
    # Previsão média entre sementes; o erro se recalcula sobre ela.
    medio = (
        df.groupby(["conjunto", "h", "modelo", "data_origem"])
        .agg(preco_prev=("preco_prev", "mean"), preco_real=("preco_real", "first"))
        .reset_index()
    )
    medio["erro"] = medio["preco_real"] - medio["preco_prev"]

    linhas = []
    for (conjunto, h), bloco in medio.groupby(["conjunto", "h"]):
        erros = bloco.pivot(index="data_origem", columns="modelo", values="erro")
        for m1, m2 in combinations(sorted(erros.columns), 2):
            par = erros[[m1, m2]].dropna()
            res = teste_dm(par[m1].to_numpy(), par[m2].to_numpy(), int(h))
            linhas.append({"conjunto": conjunto, "h": h, "modelo_1": m1, "modelo_2": m2, **res})

    tabela = pd.DataFrame(linhas, columns=list(COLUNAS[:7]))
    if tabela.empty:
        return pd.DataFrame(columns=list(COLUNAS))
    tabela["p_holm"] = tabela.groupby(["conjunto", "h"])["p_valor"].transform(holm)
    tabela["significativo"] = tabela["p_holm"] < ALFA
    tabela["melhor"] = np.where(
        tabela["estatistica"] < 0, tabela["modelo_1"], tabela["modelo_2"]
    )
    tabela.loc[~tabela["significativo"], "melhor"] = "—"
    tabela = tabela.sort_values(["conjunto", "h", "p_valor"]).reset_index(drop=True)
    return tabela[list(COLUNAS)]
