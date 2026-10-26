import numpy as np
import pandas as pd
import pytest

from tcc_cambio.avaliacao import protocolo


@pytest.fixture
def serie():
    """Série sintética de 20 dias: 10 treino, 5 validação, 5 teste."""
    n = 20
    preco = 2.0 * np.exp(np.cumsum(np.r_[0.0, np.full(n - 1, 0.01)]))
    return pd.DataFrame({
        "data": pd.bdate_range("2020-01-01", periods=n),
        "cotacao": preco,
        "log_retorno": np.r_[np.nan, np.diff(np.log(preco))],
        "conjunto": ["treino"] * 10 + ["validacao"] * 5 + ["teste"] * 5,
    })


def test_gabarito_nao_atravessa_fronteira(serie):
    gab = protocolo.gabarito(serie, "validacao", h=2)
    # validação = posições 10..14; t+2 precisa ficar <= 14
    assert gab["idx_origem"].tolist() == [10, 11, 12]
    assert (serie["conjunto"].iloc[gab["idx_origem"] + 2] == "validacao").all()


def test_gabarito_alvo_e_log_retorno_acumulado(serie):
    gab = protocolo.gabarito(serie, "teste", h=3)
    np.testing.assert_allclose(gab["y_real"], 0.03)
    np.testing.assert_allclose(gab["preco_real"], gab["preco_origem"] * np.exp(0.03))


def test_gabarito_recusa_treino(serie):
    with pytest.raises(ValueError):
        protocolo.gabarito(serie, "treino", h=1)


def test_montar_previsoes_valida_tamanho_e_nan(serie):
    gab = protocolo.gabarito(serie, "teste", h=1)
    with pytest.raises(ValueError):
        protocolo.montar_previsoes(gab, np.zeros(len(gab) - 1), "m")
    with pytest.raises(ValueError):
        protocolo.montar_previsoes(gab, np.full(len(gab), np.nan), "m")


def test_previsao_perfeita_reconstroi_o_preco(serie):
    gab = protocolo.gabaritos(serie)
    prev = protocolo.montar_previsoes(gab, gab["y_real"], "oraculo")
    df = protocolo.juntar_com_gabarito(prev, gab)
    np.testing.assert_allclose(df["preco_prev"], df["preco_real"])


def test_juntar_recusa_previsao_fora_do_protocolo(serie):
    gab = protocolo.gabarito(serie, "teste", h=1)
    prev = protocolo.montar_previsoes(gab, np.zeros(len(gab)), "m")
    prev.loc[0, "data_origem"] = serie["data"].iloc[0]  # origem no treino
    with pytest.raises(ValueError, match="fora do protocolo"):
        protocolo.juntar_com_gabarito(prev, gab)


def test_salvar_e_carregar(serie, tmp_path):
    gab = protocolo.gabaritos(serie)
    prev = protocolo.montar_previsoes(gab, np.zeros(len(gab)), "m")
    protocolo.salvar_previsoes(prev, "m", tmp_path)
    lido = protocolo.carregar_previsoes(tmp_path)
    pd.testing.assert_frame_equal(lido, prev.reset_index(drop=True), check_dtype=False)
