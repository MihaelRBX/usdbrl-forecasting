"""Passeio aleatório: o baseline ingênuo de Meese e Rogoff (1983).

Prevê que a cotação daqui a h dias é a de hoje: P̂[t+h] = P[t], ou seja,
log-retorno acumulado previsto ŷ(t, h) = 0. Não tem parâmetro nenhum.

Parece trivial, mas é o adversário mais difícil da literatura cambial: Meese
e Rogoff mostraram que modelos estruturais não o superavam fora da amostra, e
Rossi (2013) confirma que isso continua valendo em boa parte dos casos. Se
nenhum modelo do TCC vencê-lo, isso é resultado, não fracasso.

Serve também como teste de ponta a ponta do protocolo de avaliação: é o
primeiro modelo a gravar previsões e passar por `avaliacao.avaliar`.

Uso:
    python -m tcc_cambio.modelos.passeio_aleatorio
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tcc_cambio.avaliacao import protocolo

log = logging.getLogger(__name__)

NOME = "passeio_aleatorio"


def prever(serie: pd.DataFrame) -> pd.DataFrame:
    """Previsões do passeio aleatório em todos os conjuntos avaliados e horizontes."""
    gab = protocolo.gabaritos(serie)
    return protocolo.montar_previsoes(gab, np.zeros(len(gab)), modelo=NOME)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    serie = protocolo.carregar_serie_dividida()
    caminho = protocolo.salvar_previsoes(prever(serie), NOME)
    print(f"-> {caminho}")


if __name__ == "__main__":
    main()
