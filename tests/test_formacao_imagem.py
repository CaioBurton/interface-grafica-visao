"""Testes da formação de imagem (cubo, extrínsecos, K e projeções)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vc import formacao_imagem as fi  # noqa: E402

EXT = fi.PADRAO_EXTRINSECOS
INTR = fi.PADRAO_PERSPECTIVA


class TestFormacaoImagem(unittest.TestCase):
    def test_matriz_K(self):
        np.testing.assert_allclose(fi.matriz_intrinseca(800, 700, 320, 240, 2),
                                   [[800, 2, 320], [0, 700, 240], [0, 0, 1]])

    def test_rotacao_z_90(self):
        np.testing.assert_allclose(fi.rotacao_z(90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_perspectiva_inicial(self):
        v, segs = fi.projetar_cubo("perspectiva", (640, 480), EXT, INTR)
        # P1 = (-1,-1,-1) -> z=4: u = 800*(-1)/4 + 320 = 120, v = 240 - 200 = 40
        np.testing.assert_allclose(v[0], [120, 40])
        # P5 = (-1,-1,1) -> z=6: u = 320 - 133.33
        np.testing.assert_allclose(v[4], [320 - 800 / 6, 240 - 800 / 6])
        self.assertEqual(len(segs), 12)

    def test_ortografica_inicial(self):
        v, _ = fi.projetar_cubo("ortografica", (640, 480), EXT)
        np.testing.assert_allclose(v[0], [320 - 190, 240 - 190])
        np.testing.assert_allclose(v[0], v[4])  # z é descartado

    def test_atras_da_camera(self):
        v, segs = fi.projetar_cubo("perspectiva", (640, 480), {**EXT, "tz": -5}, INTR)
        self.assertTrue(np.isnan(v).all())
        self.assertEqual(segs, [])

    def test_recorte_parcial(self):
        v, segs = fi.projetar_cubo("perspectiva", (640, 480), {**EXT, "tz": 0}, INTR)
        self.assertTrue(all(np.isfinite(np.concatenate(s)).all() for s in segs))
        self.assertGreater(len(segs), 0)

    def test_renderizar(self):
        v, segs = fi.projetar_cubo("perspectiva", (800, 600), EXT, {**INTR, "cx": 400, "cy": 300})
        self.assertEqual(fi.renderizar(v, segs, (800, 600)).size, (800, 600))


if __name__ == "__main__":
    unittest.main()
