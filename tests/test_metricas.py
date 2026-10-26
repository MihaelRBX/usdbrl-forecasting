import numpy as np
import pytest

from tcc_cambio.avaliacao import metricas


def test_rmse_e_mape():
    real = np.array([2.0, 4.0])
    prev = np.array([2.2, 3.6])
    assert metricas.rmse(real, prev) == pytest.approx(np.sqrt((0.04 + 0.16) / 2))
    assert metricas.mape(real, prev) == pytest.approx(10.0)


def test_acuracia_direcional_ignora_previsao_nula():
    y_real = np.array([0.01, -0.02, 0.03, -0.01])
    y_prev = np.array([0.02, 0.01, 0.0, -0.05])
    assert metricas.acuracia_direcional(y_real, y_prev) == pytest.approx(2 / 3)


def test_acuracia_direcional_do_passeio_aleatorio_e_nan():
    assert np.isnan(metricas.acuracia_direcional(np.ones(5), np.zeros(5)))
