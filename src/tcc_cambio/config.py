from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
# Ancorados no próprio arquivo (__file__), e não no diretório de trabalho.
# Isso faz os caminhos funcionarem igual rodando da raiz do projeto, de dentro
# de notebooks/ ou de qualquer outro lugar — o caso que mais quebra na prática
# é o Jupyter, que roda com o cwd na pasta do notebook.
#
# O índice 2 reflete a estrutura src/tcc_cambio/config.py:
#     parents[0] = src/tcc_cambio/
#     parents[1] = src/
#     parents[2] = raiz do repositório
# Se este arquivo mudar de lugar, este número muda junto.
RAIZ: Final[Path] = Path(__file__).resolve().parents[2]

DIR_DADOS: Final[Path] = RAIZ / "data"
DIR_RAW: Final[Path] = DIR_DADOS / "raw"
DIR_PROCESSED: Final[Path] = DIR_DADOS / "processed"

# Saída das figuras da análise exploratória (histogramas, séries, etc.).
# Fora de data/ de propósito: são artefatos de relatório, não dados.
DIR_FIGURAS: Final[Path] = RAIZ / "reports" / "figuras"

# --------------------------------------------------------------------------
# Série do SGS / Banco Central do Brasil
# --------------------------------------------------------------------------
# O código da série é a informação mais importante deste arquivo: é o que torna
# a coleta reproduzível e é citado explicitamente na metodologia do TCC.
SERIE_PADRAO: Final[int] = 1

# TODO(confirmar): validar a descrição exata no portal do SGS antes de citar no
# texto do TCC. A série 1 é a cotação diária de venda do dólar comercial; a
# descrição abaixo precisa bater com a oficial, palavra por palavra.
SERIE_DESCRICAO: Final[str] = "Taxa de câmbio - Livre - Dólar americano (venda) - diário"

SGS_PORTAL: Final[str] = "https://www3.bcb.gov.br/sgspub/"
SGS_API: Final[str] = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"

# --------------------------------------------------------------------------
# Recorte temporal
# --------------------------------------------------------------------------
# 15 anos, escolhidos para cobrir regimes macroeconômicos distintos
# (pós-crise de 2008, recessão 2015-2016, pandemia, pós-pandemia).
ANO_INICIO: Final[int] = 2010
ANO_FIM: Final[int] = 2024

# A API do SGS recusa (HTTP 406) janelas maiores que 10 anos em séries de
# periodicidade diária, então a coleta precisa ser fatiada. 5 anos dá folga.
ANOS_POR_BLOCO: Final[int] = 5

# --------------------------------------------------------------------------
# Protocolo experimental
# --------------------------------------------------------------------------
# Divisão cronológica, sem embaralhamento: a ordem temporal precisa ser
# preservada para não vazar informação do futuro para o passado.
PROP_TREINO: Final[float] = 0.70
PROP_VALIDACAO: Final[float] = 0.15
PROP_TESTE: Final[float] = 0.15
TIMEOUT: Final[tuple[float, float]] = (5.0, 30.0)   # (conectar, ler)
TENTATIVAS: Final[int] = 4
ESPERA_BASE_S: Final[float] = 2.0

# Horizontes de previsão, em dias úteis.
HORIZONTES: Final[tuple[int, ...]] = (1, 5, 10)

# Semente base. As redes são estocásticas, então cada uma é treinada com
# N_SEMENTES execuções e as métricas são reportadas como média +/- desvio.
SEMENTE: Final[int] = 42
N_SEMENTES: Final[int] = 10
