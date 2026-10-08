"""Verificação das funções da biblioteca `vc` com casos de valor conhecido.

Execute com:  python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vc import aprendizado, camera, filtros, geometria, iluminacao, otimizacao, transformacoes  # noqa: E402

F3 = np.arange(1, 10, dtype=float).reshape(3, 3)


class TestGeometria(unittest.TestCase):
    def test_intersecao_retas(self):
        np.testing.assert_allclose(geometria.intersecao_retas([2, -1, 3], [1, 1, -5]), [2 / 3, 13 / 3])

    def test_afim(self):
        np.testing.assert_allclose(geometria.aplicar_afim([[2, 0, 1], [0, 3, -1]], [2, 1]), [5, 2])

    def test_transformar_reta(self):
        H = np.array([[1, 0.5, 2], [0, 2, -1], [0, 0.2, 1]])
        l = np.array([2, -3, 5.0])
        lt = geometria.transformar_reta(l, H)
        # um ponto sobre a reta continua sobre a reta transformada
        p_on = np.array([1.0, 7 / 3, 1.0])   # 2 - 7 + 5 = 0
        self.assertAlmostEqual(l @ p_on, 0)
        self.assertAlmostEqual(lt @ (H @ p_on), 0)
        np.testing.assert_allclose(lt, [2, -1.9091, -0.9091], atol=1e-4)

    def test_quaternion_e_rotacao(self):
        q = geometria.quaternion_eixo_angulo([0.6, 0.8, 0], 60)
        np.testing.assert_allclose(q, [0.8660, 0.3, 0.4, 0], atol=1e-4)
        R = geometria.quaternion_para_matriz(q)
        np.testing.assert_allclose(R, geometria.matriz_rotacao_eixo_angulo([0.6, 0.8, 0], 60), atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


class TestCamera(unittest.TestCase):
    def test_projecoes(self):
        np.testing.assert_allclose(camera.projecao_ortografica([3, -2, 5], 2), [6, -4])
        np.testing.assert_allclose(camera.projecao_perspectiva([4, 6, 2]), [2, 3])

    def test_fov(self):
        self.assertAlmostEqual(camera.distancia_focal_fov(36, 60), 18 / np.tan(np.deg2rad(30)))

    def test_distorcao_e_pixel(self):
        xy = camera.distorcao_radial([0.5, 0.3], 0.10, 0.05)
        np.testing.assert_allclose(xy, [0.51989, 0.31193], atol=1e-5)
        np.testing.assert_allclose(camera.para_pixel(xy, 800, 320, 240), [735.91, 489.55], atol=0.01)


class TestIluminacao(unittest.TestCase):
    def test_phong_difuso(self):
        self.assertAlmostEqual(iluminacao.phong_especular(200, 0.5, 60, 10), 100 / 1024)
        self.assertAlmostEqual(iluminacao.reflexao_difusa(200, 0.8, [0, 0, 1], [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]),
                               160 * np.sqrt(2) / 2)

    def test_lente_e_numero_f(self):
        self.assertAlmostEqual(iluminacao.lente_fina_distancia_focal(50, 25), 50 / 3)
        self.assertAlmostEqual(iluminacao.numero_f(50, 25), 2)


class TestFiltros(unittest.TestCase):
    def test_corner_separavel_centro(self):
        g = filtros.filtro_separavel(F3, filtros.H_CORNER, filtros.H_CORNER)
        self.assertAlmostEqual(g[1, 1], 0)

    def test_corner_2d_igual_separavel_no_interior(self):
        img = np.random.default_rng(0).random((8, 8))
        a = filtros.convolucao(img, filtros.nucleo_corner())[1:-1, 1:-1]
        b = filtros.filtro_separavel(img, filtros.H_CORNER, filtros.H_CORNER)[1:-1, 1:-1]
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_forma_matricial_igual_ao_filtro(self):
        Hv, Hh = filtros.matrizes_filtro_separavel(F3.shape, filtros.H_CORNER, filtros.H_CORNER)
        np.testing.assert_allclose(filtros.aplicar_matricial(F3, Hv, Hh),
                                   filtros.filtro_separavel(F3, filtros.H_CORNER, filtros.H_CORNER), atol=1e-12)

    def test_imagem_integral(self):
        s = filtros.imagem_integral(F3)
        np.testing.assert_allclose(s, [[1, 3, 6], [5, 12, 21], [12, 27, 45]])
        self.assertEqual(filtros.soma_retangulo(s, 0, 0, 1, 1), 12)
        self.assertEqual(filtros.soma_retangulo(s, 1, 1, 2, 2), 5 + 6 + 8 + 9)

    def test_media_integral_igual_convolucao(self):
        img = np.random.default_rng(1).random((9, 7))
        np.testing.assert_allclose(filtros.filtro_media_integral(img, 3),
                                   filtros.convolucao(img, filtros.nucleo_media(3)), atol=1e-12)

    def test_filtros_nao_lineares(self):
        f = np.array([[10, 2, 3], [4, 50, 6], [7, 8, 9.0]])
        W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1.0]])
        self.assertEqual(filtros.mediana(f)[1, 1], 7)
        self.assertAlmostEqual(filtros.media_alfa_cortada(f, 3, 4)[1, 1], 6.8)
        self.assertEqual(filtros.mediana_ponderada(f, W)[1, 1], 7)

    def test_mediana_remove_impulso(self):
        img = np.full((7, 7), 100.0)
        img[3, 3] = 255
        self.assertEqual(filtros.mediana(img)[3, 3], 100)

    def test_canais(self):
        rgb = np.random.default_rng(2).random((5, 6, 3))
        self.assertEqual(filtros.mediana(rgb).shape, rgb.shape)
        self.assertEqual(filtros.convolucao(rgb, filtros.nucleo_gaussiano(3)).shape, rgb.shape)


class TestTransformacoes(unittest.TestCase):
    def test_bilinear(self):
        f = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90.0]])
        self.assertAlmostEqual(transformacoes.interpolar_bilinear(f, 1, 1.5), 65)
        self.assertAlmostEqual(transformacoes.interpolar_bilinear(f, 0.5, 0.5), 30)
        self.assertEqual(transformacoes.interpolar_bilinear(f, 5, 5), 0)

    def test_escala_com_mapeamento_inverso(self):
        f = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90.0]])
        M = transformacoes.matriz_similaridade(s=2)
        x, y = transformacoes.mapeamento_inverso(M, 2, 3)
        np.testing.assert_allclose([x, y], [1, 1.5])
        g = transformacoes.transformar_imagem(f, M, saida_shape=(6, 6))
        self.assertAlmostEqual(g[3, 2], 65)

    def test_rotacao_de_90_graus(self):
        img = np.arange(16, dtype=float).reshape(4, 4)
        M = transformacoes.similaridade_centrada(img.shape, 1, 90)
        np.testing.assert_allclose(transformacoes.transformar_imagem(img, M), np.rot90(img, -1), atol=1e-9)

    def test_reamostragem(self):
        f = np.array([[10, 20], [30, 40.0]])
        up = transformacoes.aumentar_resolucao(f, 2)
        self.assertEqual(up.shape, (4, 4))
        self.assertAlmostEqual(up[1, 1], 25)
        np.testing.assert_allclose(transformacoes.diminuir_resolucao(f, 2), [[11.25]])

    def test_over(self):
        self.assertAlmostEqual(transformacoes.composicao_over(200, 100, 0.3), 130)

    def test_rbf(self):
        w, v = transformacoes.rbf_interpolacao([0, 1, 2], [1, 3, 2], 1.8)
        np.testing.assert_allclose(w, [1.75, -1.5, 1.25])
        self.assertAlmostEqual(v, 2.2)
        np.testing.assert_allclose(transformacoes.rbf_interpolar([0, 1, 2], [0, 1, 2], w), [1, 3, 2], atol=1e-12)

    def test_distorcao_nula_nao_altera(self):
        img = np.random.default_rng(3).random((10, 12))
        np.testing.assert_allclose(transformacoes.corrigir_distorcao_radial(img, 0, 0), img, atol=1e-9)


class TestOtimizacao(unittest.TestCase):
    def test_energias(self):
        self.assertEqual(otimizacao.energia_dados([[10, 20], [30, 40]], [[12, 18], [29, 43]]), 18)
        self.assertEqual(otimizacao.energia_suavidade([10, 12, 15]), 13)

    def test_map(self):
        x = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        y = [1, 1, 0, 1, 0, 0, 0, 0, 1]
        rot, post = otimizacao.map_por_frequencia(x, y, 0)
        self.assertEqual(rot, 0)
        np.testing.assert_allclose(post, [4 / 9, 1 / 9])

    def test_mrf(self):
        e0 = otimizacao.energia_rotulo_mrf(0, 0, [1, 1, 1, 1], w=1, s=0.2)
        e1 = otimizacao.energia_rotulo_mrf(1, 0, [1, 1, 1, 1], w=1, s=0.2)
        self.assertAlmostEqual(e0, 0.8)
        self.assertAlmostEqual(e1, 1.0)

    def test_icm_remove_ruido(self):
        img = np.ones((9, 9))
        img[4, 4] = 0
        out = otimizacao.restaurar_icm(img, s=0.4)
        self.assertEqual(out.min(), 1)


class TestAprendizado(unittest.TestCase):
    X = [(1, 1), (2, 1), (1, 2), (4, 4), (5, 5), (4, 5)]
    y = ["A", "A", "A", "B", "B", "B"]

    def test_knn(self):
        self.assertEqual(aprendizado.knn_classificar(self.X, self.y, (3, 3), 1)[0], "B")
        self.assertEqual(aprendizado.knn_classificar(self.X, self.y, (3, 3), 3)[0], "B")
        self.assertEqual(aprendizado.knn_classificar(self.X, self.y, (1.2, 1.2), 3)[0], "A")

    def test_logistica(self):
        self.assertAlmostEqual(aprendizado.regressao_logistica_prob([1, -1], 0, [2, 1]), 0.7310586, places=6)

    def test_svm(self):
        X = [(1, 4), (2, 2), (3, 3), (0, 0), (0, 1), (2, 0)]
        y = [1, 1, 1, -1, -1, -1]
        sv = aprendizado.svm_vetores_suporte([1, 1], -3, X, y)
        self.assertEqual(sorted(sv.tolist()), [1, 5])
        self.assertEqual(aprendizado.svm_classificar([1, 1], -3, [[2, 1.1]])[0], 1)

    def test_kmeans(self):
        P = [(1, 1), (2, 1), (4, 4), (5, 5)]
        C, rot = aprendizado.kmeans(P, centroides=[(4, 3), (6, 6)], iteracoes=2)
        np.testing.assert_allclose(C, [[1.5, 1], [4.5, 4.5]])
        self.assertEqual(rot.tolist(), [0, 0, 1, 1])
        C1, _ = aprendizado.kmeans(P, centroides=[(4, 3), (6, 6)], iteracoes=1)
        np.testing.assert_allclose(C1, [[7 / 3, 2], [5, 5]])

    def test_gmm(self):
        X = [(1, 0), (2, 1), (3, 3), (-1, -1)]
        g = aprendizado.gmm_passo_e(X, [0.4, 0.6], [[0, 0], [2, 2]], [np.eye(2), [[2, 0], [0, 1]]])
        np.testing.assert_allclose(g.sum(1), 1)
        self.assertEqual(g.argmax(1).tolist(), [0, 1, 1, 0])
        np.testing.assert_allclose(g[0], [0.8444, 0.1556], atol=1e-4)

    def test_pca(self):
        media, C, vals, vecs = aprendizado.pca([(2, 0), (0, 2), (3, 1), (1, 3)])
        np.testing.assert_allclose(media, [1.5, 1.5])
        np.testing.assert_allclose(C, [[1.25, -0.75], [-0.75, 1.25]])
        np.testing.assert_allclose(vals, [2, 0.5])
        self.assertAlmostEqual(aprendizado.variancia_explicada(vals)[0], 0.8)

    def test_segmentacao(self):
        img = np.zeros((6, 6, 3))
        img[:, 3:] = 255
        seg = aprendizado.segmentar_imagem_kmeans(img, k=2)
        np.testing.assert_allclose(seg, img)


if __name__ == "__main__":
    unittest.main()
