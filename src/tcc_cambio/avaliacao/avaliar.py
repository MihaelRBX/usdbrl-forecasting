"""Avaliação comparativa: junta as previsões de todos os modelos e gera as tabelas.

Lê `data/resultados/previsoes_*.csv` (um arquivo por modelo, ver
`avaliacao.protocolo`), casa com o gabarito, calcula RMSE/MAPE/acurácia
direcional e roda Diebold-Mariano entre todos os pares.

Uso:
    python -m tcc_cambio.avaliacao.avaliar
    python -m tcc_cambio.avaliacao.avaliar --conjunto teste
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from tcc_cambio.avaliacao import diebold_mariano, metricas, protocolo
from tcc_cambio.config import DIR_RESULTADOS, DIR_TABELAS

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas e Diebold-Mariano de todos os modelos.")
    parser.add_argument("--resultados", type=Path, default=DIR_RESULTADOS,
                        help="diretório com os previsoes_*.csv")
    parser.add_argument("--saida", type=Path, default=DIR_TABELAS,
                        help="diretório das tabelas geradas")
    parser.add_argument("--conjunto", choices=protocolo.CONJUNTOS_AVALIADOS,
                        help="restringe a um conjunto (padrão: ambos)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 140)

    serie = protocolo.carregar_serie_dividida()
    prev = protocolo.carregar_previsoes(args.resultados)
    if args.conjunto:
        prev = prev[prev["conjunto"] == args.conjunto]
    df = protocolo.juntar_com_gabarito(prev, protocolo.gabaritos(serie))

    tab_metricas = metricas.tabela_metricas(df)
    tab_dm = diebold_mariano.comparar_todos(df)

    args.saida.mkdir(parents=True, exist_ok=True)
    tab_metricas.to_csv(args.saida / "metricas.csv", index=False)
    tab_dm.to_csv(args.saida / "diebold_mariano.csv", index=False)

    print("\nMétricas (preço reconstruído; média entre sementes):")
    print(tab_metricas.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if tab_dm.empty:
        print("\nDiebold-Mariano: só há um modelo — nada a comparar ainda.")
    else:
        print("\nDiebold-Mariano (HLN, p-valor ajustado por Holm):")
        print(tab_dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n-> {args.saida / 'metricas.csv'}\n-> {args.saida / 'diebold_mariano.csv'}")


if __name__ == "__main__":
    main()
