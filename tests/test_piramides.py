"""Testes das pirâmides e do blending (python -m unittest discover -s tests)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vc import piramides  # noqa: E402

RNG = np.random.default_rng(0)


class TestPiramides(unittest.TestCase):
    def test_reconstrucao_exata(self):
        img = RNG.random((64, 48, 3)) * 255
        np.testing.assert_allclose(piramides.colapsar(piramides.piramide_laplaciana(img, 4)), img, atol=1e-9)

    def test_tamanhos(self):
        g = piramides.piramide_gaussiana(np.zeros((64, 64)), 4)
        self.assertEqual([x.shape for x in g], [(64, 64), (32, 32), (16, 16), (8, 8)])

    def test_blending_mascara_constante(self):
        a, b = RNG.random((32, 32, 3)) * 255, RNG.random((32, 32, 3)) * 255
        np.testing.assert_allclose(piramides.blending(a, b, np.ones((32, 32)), 3), a, atol=1e-9)
        np.testing.assert_allclose(piramides.blending(a, b, np.zeros((32, 32)), 3), b, atol=1e-9)

    def test_blending_suaviza_emenda(self):
        a, b = np.full((32, 32), 200.0), np.full((32, 32), 50.0)
        m = np.zeros((32, 32)); m[:, 16:] = 1
        salto = lambda im: np.abs(np.diff(im, axis=1)).max()
        self.assertLess(salto(piramides.blending(a, b, m, 4)), salto(piramides.justaposicao(a, b, m)))


if __name__ == "__main__":
    unittest.main()
