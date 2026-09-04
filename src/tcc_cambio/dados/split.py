"""Divisão cronológica da série USD/BRL em treino, validação e teste.

Split **cronológico e sem embaralhamento**: a ordem temporal precisa ser
preservada para não vazar informação do futuro para o passado. Proporções em
`config.py` — 70% / 15% / 15%.

Estratégia de saída: um único CSV com a coluna `conjunto` rotulando cada
observação, em vez de três arquivos separados. Duas razões:

1. **Fonte única de verdade.** A divisão fica documentada no próprio dado, e
   quem consome filtra: `df[df.conjunto == "treino"]`.

2. **Janelas deslizantes atravessam a fronteira.** A primeira janela da
   validação precisa dos últimos N dias do treino como entrada. Isso **não é
   vazamento** — olhar para o passado é legítimo; vazamento seria ajustar
   parâmetros (escala, limiar de codificação) usando dados de validação ou
   teste. Com um arquivo só, montar essas janelas é natural; com três, você
   perderia N observações em cada fronteira.

Uso:
    python -m tcc_cambio.dados.split
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from tcc_cambio.config import (
    DIR_PROCESSED,
    PROP_TESTE,
    PROP_TREINO,
    PROP_VALIDACAO,
)

log = logging.getLogger(__name__)

ROTULOS = ("treino", "validacao", "teste")


def calcular_fronteiras(n: int, prop_treino: float, prop_validacao: float) -> tuple[int, int]:
    """Devolve os dois índices de corte da série.

    O retorno (i, j) significa:
        treino    = linhas [0, i)
        validação = linhas [i, j)
        teste     = linhas [j, n)

    Cuidado: as três proporções raramente dão números inteiros. A soma dos três
    conjuntos precisa ser exatamente `n` — nenhuma observação pode se perder.
    """
    obs_treino = int(n * prop_treino)
    obs_validacao = int(n * prop_validacao)

    return obs_treino, obs_treino + obs_validacao


def rotular(df: pd.DataFrame, prop_treino: float, prop_validacao: float) -> pd.DataFrame:
    """Acrescenta a coluna `conjunto` com os rótulos de ROTULOS.

    Não reordena e não remove nada: apenas rotula, respeitando a ordem
    cronológica já existente no DataFrame.
    """
    df = df.copy()

    n = len(df)

    i, j = calcular_fronteiras(n, prop_treino, prop_validacao)

    rotulos = [ROTULOS[0]]*i + [ROTULOS[1]]*(j-i) + [ROTULOS[2]]*(n-j) 

    df["conjunto"] = rotulos

    return df




def validar_split(df: pd.DataFrame) -> None:
    """Verifica as invariantes da divisão. Levanta na primeira violação.

    O que precisa ser verdade:
      - os três conjuntos existem e nenhum está vazio;
      - a soma dos tamanhos é igual ao total de linhas;
      - os blocos são contíguos no tempo (treino inteiro antes da validação,
        que vem inteira antes do teste) — nada de rótulo intercalado.
    """
    if "conjunto" not in df.columns:
        raise ValueError("coluna 'conjunto' ausente — rode rotular() antes.")

    tamanhos = df["conjunto"].value_counts()

    vazios = [r for r in ROTULOS if tamanhos.get(r, 0) == 0]
    if vazios:
        raise ValueError(f"conjunto(s) vazio(s) ou ausente(s): {vazios} (n={len(df)})")

    inesperados = sorted(set(tamanhos.index) - set(ROTULOS))
    if inesperados:
        raise ValueError(f"rótulos inesperados em 'conjunto': {inesperados}")

    if int(tamanhos.sum()) != len(df):
        raise ValueError(f"soma dos conjuntos ({int(tamanhos.sum())}) difere do total ({len(df)})")

    ordem = df["conjunto"].map({r: k for k, r in enumerate(ROTULOS)}).reset_index(drop=True)
    if not ordem.is_monotonic_increasing:
        pos = int((ordem.diff() < 0).idxmax())
        raise ValueError(
            f"conjuntos não são contíguos: na posição {pos} vem "
            f"'{df['conjunto'].iloc[pos]}' logo após '{df['conjunto'].iloc[pos - 1]}'. "
            "Split fora de ordem cronológica implica vazamento temporal."
        )



def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Loga tamanho, proporção real e intervalo de datas de cada conjunto.

    As proporções realizadas diferem levemente das nominais por causa do
    arredondamento — registrar isso é material direto para a metodologia do TCC.
    """

    linhas = []

    for rotulo in ROTULOS:
        bloco = df[df["conjunto"] == rotulo]
        linhas.append({
            "conjunto": rotulo,
            "obs": len(bloco),
            "proporcao": len(bloco)/len(df),
            "inicio": bloco["data"].iloc[0].date(),
            "fim": bloco["data"].iloc[-1].date(),
            "cotacao_min": bloco["cotacao"].min() ,
            "cotacao_max": bloco["cotacao"].max()
        })

    resumo = pd.DataFrame(linhas)

    log.info("resumo do split:\n%s \n", resumo.to_string(index=False))

    return resumo


    


def salvar(df: pd.DataFrame, caminho: Path) -> Path:
    """Grava a série rotulada em CSV com datas em ISO-8601."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, date_format="%Y-%m-%d")
    log.info("gravado %s (%d linhas)", caminho.name, len(df))
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide a série USD/BRL em treino/validação/teste, cronologicamente."
    )
    parser.add_argument("--entrada", type=Path,
                        default=DIR_PROCESSED / "usdbrl_diario.csv",
                        help="CSV da série tratada")
    parser.add_argument("--saida", type=Path,
                        default=DIR_PROCESSED / "usdbrl_split.csv",
                        help="CSV de saída, com a coluna conjunto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.entrada.exists():
        parser.error(f"arquivo não encontrado: {args.entrada}\n"
                     "Rode antes: python -m tcc_cambio.dados.tratamento")

    df = pd.read_csv(args.entrada, parse_dates=["data"])

    df = rotular(df, PROP_TREINO, PROP_VALIDACAO)
    validar_split(df)
    resumir(df)
    caminho = salvar(df, args.saida)

    print(f"\n{len(df)} observações de {df['data'].iloc[0].date()} "
          f"a {df['data'].iloc[-1].date()}")
    print(f"-> {caminho}")


if __name__ == "__main__":
    main()
