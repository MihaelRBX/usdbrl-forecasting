"""Histogramas da série USD/BRL: distribuição da cotação e do log-retorno.

Primeira peça da análise exploratória (metodologia, §3.2). Duas distribuições
interessam ao TCC, por motivos diferentes:

1. **Cotação (nível de preço).** Série não-estacionária, com tendência e
   multimodalidade que refletem os regimes cambiais do período (2010–2024).
   O histograma serve para *mostrar* o problema — não é o que se modela.

2. **Log-retorno diário.** É o alvo estacionário do estudo: o que o ARMA-GARCH
   modela e o que a codificação por eventos da SNN representa. O histograma,
   sobreposto à normal de mesma média e desvio, torna visível o fato central da
   série cambial — **caudas pesadas e excesso de curtose** (leptocurtose), a
   justificativa empírica para o componente GARCH. Assimetria e curtose são
   anotadas na figura porque é o tipo de rigor descritivo que o baseline
   econométrico exige.

Uso:
    python -m tcc_cambio.analise.histogramas
    python -m tcc_cambio.analise.histogramas --bins 60
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem display: roda em terminal e em CI.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcc_cambio.config import DIR_FIGURAS, DIR_PROCESSED

log = logging.getLogger(__name__)

# Número de classes padrão. 50 dá resolução suficiente em ~3.767 observações
# sem picotar a distribuição em ruído.
BINS_PADRAO = 50


def carregar_serie(caminho: Path) -> pd.DataFrame:
    """Lê a série canônica tratada (saída de `dados.tratamento`)."""
    df = pd.read_csv(caminho, parse_dates=["data"])
    faltando = {"cotacao", "log_retorno"} - set(df.columns)
    if faltando:
        raise ValueError(
            f"colunas ausentes em {caminho.name}: {faltando}. "
            "Rode antes: python -m tcc_cambio.dados.tratamento"
        )
    log.info("carregada série tratada: %d observações de %s a %s",
             len(df), df["data"].iloc[0].date(), df["data"].iloc[-1].date())
    return df


def _densidade_normal(x: np.ndarray, media: float, desvio: float) -> np.ndarray:
    """Densidade da normal N(media, desvio²) avaliada em x."""
    return (1.0 / (desvio * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - media) / desvio) ** 2
    )


def histograma_cotacao(df: pd.DataFrame, bins: int, caminho: Path) -> Path:
    """Histograma do nível de preço (R$/US$). Documenta a não-estacionariedade."""
    cot = df["cotacao"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cot, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.4)
    ax.axvline(cot.mean(), color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"média = R$ {cot.mean():.2f}")
    ax.axvline(np.median(cot), color="#27ae60", linestyle=":", linewidth=1.2,
               label=f"mediana = R$ {np.median(cot):.2f}")

    ax.set_title("Distribuição da cotação diária USD/BRL (2010–2024)")
    ax.set_xlabel("Cotação (BRL por USD)")
    ax.set_ylabel("Frequência (dias úteis)")
    ax.legend(frameon=False)
    fig.tight_layout()

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    log.info("gravado %s", caminho.name)
    return caminho


def histograma_log_retorno(df: pd.DataFrame, bins: int, caminho: Path) -> Path:
    """Histograma do log-retorno diário com normal sobreposta.

    Anota assimetria e curtose — o leptocurtismo é a evidência empírica que
    motiva o GARCH.
    """
    ret = df["log_retorno"].dropna().to_numpy()
    media, desvio = ret.mean(), ret.std(ddof=1)

    # Assimetria e curtose amostrais (curtose "em excesso": normal = 0).
    z = (ret - media) / desvio
    assimetria = float(np.mean(z ** 3))
    curtose_excesso = float(np.mean(z ** 4) - 3.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ret, bins=bins, density=True, color="#3b6ea5",
            edgecolor="white", linewidth=0.4, label="log-retorno observado")

    # Normal de mesma média e desvio: a referência da qual a série se afasta.
    grade = np.linspace(ret.min(), ret.max(), 400)
    ax.plot(grade, _densidade_normal(grade, media, desvio),
            color="#c0392b", linewidth=1.6,
            label=f"normal N({media:.4f}, {desvio:.4f}²)")

    caixa = (f"n = {ret.size}\n"
             f"média = {media:+.5f}\n"
             f"desvio = {desvio:.5f}\n"
             f"assimetria = {assimetria:+.3f}\n"
             f"curtose (exc.) = {curtose_excesso:+.3f}")
    ax.text(0.02, 0.97, caixa, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                      edgecolor="0.7"))

    ax.set_title("Distribuição do log-retorno diário USD/BRL (2010–2024)")
    ax.set_xlabel("Log-retorno diário  log(P_t) − log(P_{t−1})")
    ax.set_ylabel("Densidade")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    log.info("gravado %s  (assimetria %+.3f | curtose exc. %+.3f)",
             caminho.name, assimetria, curtose_excesso)
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera os histogramas da cotação e do log-retorno USD/BRL."
    )
    parser.add_argument("--entrada", type=Path,
                        default=DIR_PROCESSED / "usdbrl_diario.csv",
                        help="CSV da série tratada")
    parser.add_argument("--saida", type=Path, default=DIR_FIGURAS,
                        help="diretório para as figuras")
    parser.add_argument("--bins", type=int, default=BINS_PADRAO,
                        help="número de classes dos histogramas")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.entrada.exists():
        parser.error(f"arquivo não encontrado: {args.entrada}\n"
                     "Rode antes: python -m tcc_cambio.dados.tratamento")

    df = carregar_serie(args.entrada)
    p_cot = histograma_cotacao(df, args.bins, args.saida / "hist_cotacao.png")
    p_ret = histograma_log_retorno(df, args.bins,
                                   args.saida / "hist_log_retorno.png")

    print("\nHistogramas gerados:")
    print(f"  -> {p_cot}")
    print(f"  -> {p_ret}")


if __name__ == "__main__":
    main()
