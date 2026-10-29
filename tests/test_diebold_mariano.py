import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcc_cambio.avaliacao import diebold_mariano as dm


@pytest.fixture
def erros():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1.0, 300), rng.normal(0, 1.2, 300)


def test_h1_equivale_ao_teste_t_do_diferencial(erros):
    # Com h = 1, a correção HLN transforma o DM no teste t de uma amostra sobre d.
    e1, e2 = erros
    res = dm.teste_dm(e1, e2, h=1)
    t = stats.ttest_1samp(e1**2 - e2**2, 0.0)
    assert res["estatistica"] == pytest.approx(t.statistic)
    assert res["p_valor"] == pytest.approx(t.pvalue)


def test_antissimetria(erros):
    e1, e2 = erros
    a, b = dm.teste_dm(e1, e2, h=5), dm.teste_dm(e2, e1, h=5)
    assert a["estatistica"] == pytest.approx(-b["estatistica"])
    assert a["p_valor"] == pytest.approx(b["p_valor"])


def test_previsoes_identicas_dao_nan(erros):
    e1, _ = erros
    assert np.isnan(dm.teste_dm(e1, e1, h=1)["estatistica"])


def test_detecta_modelo_claramente_melhor():
    rng = np.random.default_rng(1)
    res = dm.teste_dm(rng.normal(0, 0.5, 500), rng.normal(0, 2.0, 500), h=1)
    assert res["estatistica"] < 0 and res["p_valor"] < 0.001


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"])
    np.testing.assert_allclose(dm.holm(p).to_numpy(), [0.03, 0.06, 0.06])


def _juntado(modelos):
    """Previsões já juntadas ao gabarito, para `comparar_todos`."""
    rng = np.random.default_rng(2)
    datas = pd.bdate_range("2024-01-01", periods=50)
    real = 5 + rng.normal(0, 0.05, 50)
    return pd.concat([
        pd.DataFrame({"conjunto": "teste", "h": 1, "modelo": m, "semente": -1,
                      "data_origem": datas, "preco_real": real,
                      "preco_prev": real + rng.normal(0, 0.01 * (k + 1), 50)})
        for k, m in enumerate(modelos)
    ])


def test_comparar_todos_com_um_modelo_tem_cabecalho():
    tabela = dm.comparar_todos(_juntado(["a"]))
    assert tabela.empty
    assert tuple(tabela.columns) == dm.COLUNAS


def test_comparar_todos_com_dois_modelos():
    tabela = dm.comparar_todos(_juntado(["a", "b"]))
    assert len(tabela) == 1
    assert tuple(tabela.columns) == dm.COLUNAS
    assert tabela.loc[0, "melhor"] == "a"
