"""Coleta da série USD/BRL do SGS/Banco Central do Brasil.

Baixa a série bruta e persiste sem transformação alguma, junto de um manifesto
com os metadados da coleta. Toda limpeza acontece depois, em tratamento.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import requests

from tcc_cambio.config import (
    ANO_FIM,
    ANO_INICIO,
    ANOS_POR_BLOCO,
    DIR_RAW,
    SERIE_DESCRICAO,
    SERIE_PADRAO,
    SGS_API,
    TIMEOUT,
    TENTATIVAS,
    ESPERA_BASE_S,

)

log = logging.getLogger(__name__)


def montar_blocos(ano_inicio: int, ano_fim: int, anos_por_bloco: int = ANOS_POR_BLOCO
                  ) -> list[tuple[date, date]]:
    """Fatia o intervalo em janelas aceitas pela API (máx. 10 anos em série diária, limitacao da API).

    Fronteiras inclusivas nos dois lados: (2010-01-01, 2014-12-31) tem 5 anos.
    """
    blocos = []
    ano = ano_inicio

    while ano <= ano_fim:
        ultimo = min(ano + anos_por_bloco - 1, ano_fim)
        blocos.append((date(ano, 1, 1), date(ultimo, 12, 31)))
        ano = ultimo + 1

    return blocos


def buscar_bloco(serie: int, inicio: date, fim: date) -> list[dict]:
    """Requisita um bloco. Repete em falha transitória; falha na hora em erro 4xx."""

    params = {
        "formato": "json",
        "dataInicial": inicio.strftime("%d/%m/%Y"),
        "dataFinal": fim.strftime("%d/%m/%Y")
    }

    ultimo_erro: Exception | None = None

    for tentativa in range(1, TENTATIVAS + 1):
        try:
            url = SGS_API.format(serie=serie)
            r = requests.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 406:
                raise RuntimeError(
                    f"HTTP 406 no bloco {inicio}..{fim}: janela maior que 10 anos "
                    f"ou parâmetros ausentes — verifique ANOS_POR_BLOCO."
                )

            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"HTTP {r.status_code} no bloco {inicio}..{fim}: {r.text[:200]}")

            r.raise_for_status()

            dados = r.json()
            if not isinstance(dados, list): 
                raise ValueError(f"resposta não é uma lista: {dados!r:.200}")

            log.info("bloco %s a %s -> %d registros", inicio, fim, len(dados))

            return dados
        
        except(requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            ultimo_erro = exc
            if tentativa < TENTATIVAS:
                espera = ESPERA_BASE_S * 2 ** (tentativa - 1)
                log.warning("bloco %s..%s falhou (%s). Nova tentativa em %.0fs",
                            inicio, fim, exc, espera)
                time.sleep(espera)

    raise RuntimeError(
        f"falha ao buscar o bloco {inicio}..{fim} após {TENTATIVAS} tentativas"
    ) from ultimo_erro


def coletar(serie: int, ano_inicio: int, ano_fim: int) -> list[dict]:
    """Baixa a série inteira: itera blocos, concatena, deduplica e ordena."""
    blocos = montar_blocos(ano_inicio, ano_fim, ANOS_POR_BLOCO)

    registros = []          
    vistos = set()

    for inicio, fim in blocos:
        dados = buscar_bloco(serie, inicio, fim)
        for registro in dados:
            data = registro["data"]
            if data not in vistos:
                vistos.add(data)
                registros.append(registro)
        time.sleep(0.5)

    registros.sort(key=lambda r: datetime.strptime(r["data"], "%d/%m/%Y"))

    return registros





def salvar(registros: list[dict], serie: int, ano_inicio: int, ano_fim: int,
           dir_saida: Path) -> Path:
    """Grava o JSON bruto e o manifesto de proveniência ao lado.

    Os dois arquivos formam um artefato único — dado e procedência. Gravá-los
    na mesma função garante que nunca divirjam.
    """
    if not registros:
        raise ValueError("nada a gravar: a lista de registros está vazia.")

    dir_saida.mkdir(parents=True, exist_ok=True)
    nome = f"sgs_{serie}_{ano_inicio}_{ano_fim}"
    caminho_json = dir_saida / f"{nome}.json"

    # indent=2 porque os dados são versionados no git: sem indentação, o diff
    # de qualquer alteração seria uma única linha ilegível.
    conteudo = json.dumps(registros, ensure_ascii=False, indent=2)
    caminho_json.write_text(conteudo, encoding="utf-8")

    manifesto = {
        "serie_sgs": serie,
        "descricao": SERIE_DESCRICAO,
        "fonte": SGS_API.format(serie=serie),
        "periodo_solicitado": {
            "inicio": f"{ano_inicio}-01-01",
            "fim": f"{ano_fim}-12-31",
        },
        # Difere do solicitado quando as pontas caem em feriado ou fim de semana.
        # Registrar os dois documenta que a diferença é esperada, não perda.
        "periodo_recebido": {
            "primeira_data": registros[0]["data"],
            "ultima_data": registros[-1]["data"],
        },
        "n_registros": len(registros),
        # Com fuso: sem astimezone() o carimbo fica ambíguo.
        "coletado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        # Hash da MESMA string gravada em disco — só assim ele descreve o arquivo.
        "sha256": hashlib.sha256(conteudo.encode("utf-8")).hexdigest(),
        "arquivo": caminho_json.name,
    }

    caminho_manifesto = dir_saida / f"{nome}.manifesto.json"
    caminho_manifesto.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("gravado %s (%d registros)", caminho_json.name, len(registros))
    return caminho_json


def main() -> None:
    """Ponto de entrada: lê a linha de comando, executa o pipeline e relata."""
    parser = argparse.ArgumentParser(
        description="Coleta a série diária USD/BRL do SGS/Banco Central do Brasil."
    )
    parser.add_argument("--serie", type=int, default=SERIE_PADRAO,
                        help=f"código da série no SGS (padrão: {SERIE_PADRAO})")
    parser.add_argument("--inicio", type=int, default=ANO_INICIO,
                        help=f"ano inicial, inclusivo (padrão: {ANO_INICIO})")
    parser.add_argument("--fim", type=int, default=ANO_FIM,
                        help=f"ano final, inclusivo (padrão: {ANO_FIM})")
    parser.add_argument("--saida", type=Path, default=DIR_RAW,
                        help="diretório de destino (padrão: data/raw)")
    args = parser.parse_args()

    if args.inicio > args.fim:
        parser.error(f"--inicio ({args.inicio}) não pode ser maior que --fim ({args.fim})")

    # Configuração de logging pertence à aplicação, nunca ao módulo importável.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    registros = coletar(args.serie, args.inicio, args.fim)
    caminho = salvar(registros, args.serie, args.inicio, args.fim, args.saida)

    # Resultado vai para stdout; progresso e avisos ficam no stderr (logging).
    print(f"\n{len(registros)} registros de "
          f"{registros[0]['data']} a {registros[-1]['data']}")
    print(f"-> {caminho}")


if __name__ == "__main__":
    main()
