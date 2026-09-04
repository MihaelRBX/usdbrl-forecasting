# Previsão da Taxa de Câmbio USD/BRL

Trabalho de Conclusão de Curso (Ciência da Computação — Universidade Presbiteriana
Mackenzie). Estudo comparativo entre um modelo econométrico clássico
(**ARMA-GARCH**) e arquiteturas de redes neurais (**LSTM**, **GRU** e **Spiking
Neural Networks**) na previsão da taxa de câmbio **USD/BRL**, a partir de dados
históricos do Banco Central do Brasil.

- **Autores:** Mihael Rommel Barbosa Xavier · Edson Fu · Rafael Moutinho
- **Orientador pretendido:** Prof. Anderson Adaime de Borba
- **Execução prevista:** Ago/2026 – Jul/2027

O eixo do trabalho é *como cada modelo representa o tempo*: da estrutura
autorregressiva do ARMA-GARCH à codificação por eventos das SNN.

## Estrutura do repositório

```
src/tcc_cambio/        Pacote Python instalável
  config.py            Caminhos, série do SGS, recorte temporal e protocolo experimental
  dados/
    coleta_sgs.py      Coleta da série USD/BRL do SGS/BCB (bruta + manifesto)
    tratamento.py      Limpeza e validação: JSON cru -> série canônica
    split.py           Divisão cronológica em treino/validação/teste
  analise/
    histogramas.py     Análise exploratória (cotação e log-retorno)
data/
  raw/                 Artefatos imutáveis da coleta (JSON + manifesto)
  processed/           Série canônica (usdbrl_diario.csv) e divisão (usdbrl_split.csv)
reports/figuras/       Figuras geradas pela análise
notebooks/  tests/      (reservados)
```

A documentação do TCC (base de conhecimento, metodologia, referências e entregas
da disciplina) vive em um repositório separado: **`tcc-cambio-docs`**. Este repo
guarda apenas o projeto reproduzível — código, dados e saídas.

> `data/` é versionado **de propósito**: a série tem ~100 KB e a reprodutibilidade
> é requisito metodológico do TCC — `git clone` deve bastar para reproduzir o
> experimento. Ver a metodologia no repositório `tcc-cambio-docs`.

## Instalação

Requer Python ≥ 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # ou: pip install -r requirements.txt
```

Grupos opcionais de dependências (`pyproject.toml`): `econometria`
(statsmodels, arch), `redes` (torch, snntorch), `dev` (pytest, ruff, jupyter,
matplotlib).

## Reprodução do pipeline

```bash
python -m tcc_cambio.dados.coleta_sgs      # baixa a série bruta do SGS -> data/raw/
python -m tcc_cambio.dados.tratamento      # trata e valida        -> data/processed/
python -m tcc_cambio.dados.split           # divide treino/val/teste -> data/processed/
python -m tcc_cambio.analise.histogramas   # gera figuras          -> reports/figuras/
```

## Dados

Série 1 do SGS/BCB — *Taxa de câmbio — Livre — Dólar americano (venda) — diário*,
recorte **2010–2026**, escolhido para cobrir regimes macroeconômicos distintos
(pós-crise de 2008, recessão de 2015–2016, pandemia e pós-pandemia).
