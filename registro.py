"""Catálogo das funções numéricas da biblioteca `vc` exibidas na aba "Funções" da interface.

Cada entrada: (categoria, nome, função, [(parâmetro, valor padrão em sintaxe Python), ...]).
"""
from vc import aprendizado, camera, filtros, geometria, iluminacao, otimizacao, transformacoes

F3 = "[[1,2,3],[4,5,6],[7,8,9]]"

FUNCOES = [
    ("Geometria", "intersecao_retas", geometria.intersecao_retas, [("a", "[2,-1,3]"), ("b", "[1,1,-5]")]),
    ("Geometria", "reta_por_pontos", geometria.reta_por_pontos, [("p", "[0,0]"), ("q", "[1,1]")]),
    ("Geometria", "aplicar_afim", geometria.aplicar_afim, [("A", "[[2,0,1],[0,3,-1]]"), ("p", "[2,1]")]),
    ("Geometria", "aplicar_projetiva", geometria.aplicar_projetiva,
     [("H", "[[1,0.5,2],[0,2,-1],[0,0.2,1]]"), ("p", "[2,1]")]),
    ("Geometria", "transformar_reta", geometria.transformar_reta,
     [("l", "[2,-3,5]"), ("H", "[[1,0.5,2],[0,2,-1],[0,0.2,1]]")]),
    ("Geometria", "quaternion_eixo_angulo", geometria.quaternion_eixo_angulo,
     [("n", "[0.6,0.8,0]"), ("theta_graus", "60")]),
    ("Geometria", "quaternion_para_matriz", geometria.quaternion_para_matriz, [("q", "[0.866,0.3,0.4,0]")]),
    ("Geometria", "matriz_rotacao_eixo_angulo", geometria.matriz_rotacao_eixo_angulo,
     [("n", "[0.6,0.8,0]"), ("theta_graus", "60")]),

    ("Câmera", "projecao_ortografica", camera.projecao_ortografica, [("p", "[3,-2,5]"), ("s", "2")]),
    ("Câmera", "projecao_perspectiva", camera.projecao_perspectiva, [("p", "[4,6,2]"), ("f", "1")]),
    ("Câmera", "distancia_focal_fov", camera.distancia_focal_fov, [("largura_sensor", "36"), ("fov_graus", "60")]),
    ("Câmera", "distorcao_radial", camera.distorcao_radial, [("xy", "[0.5,0.3]"), ("k1", "0.1"), ("k2", "0.05")]),
    ("Câmera", "para_pixel", camera.para_pixel, [("xy", "[0.5,0.3]"), ("f", "800"), ("cx", "320"), ("cy", "240")]),

    ("Iluminação", "phong_especular", iluminacao.phong_especular,
     [("Li", "200"), ("ks", "0.5"), ("theta_s_graus", "60"), ("ke", "10")]),
    ("Iluminação", "reflexao_difusa", iluminacao.reflexao_difusa,
     [("Li", "200"), ("kd", "0.8"), ("n", "[0,0,1]"), ("v", "[1,0,1]")]),
    ("Iluminação", "lente_fina_distancia_focal", iluminacao.lente_fina_distancia_focal, [("z0", "50"), ("zi", "25")]),
    ("Iluminação", "numero_f", iluminacao.numero_f, [("f", "50"), ("d", "25")]),
    ("Iluminação", "composicao_over", transformacoes.composicao_over,
     [("frente", "200"), ("fundo", "100"), ("alpha", "0.3")]),

    ("Filtros (matriz)", "convolucao", filtros.convolucao, [("img", F3), ("nucleo", "[[0,1,0],[1,-4,1],[0,1,0]]")]),
    ("Filtros (matriz)", "filtro_separavel", filtros.filtro_separavel,
     [("img", F3), ("h_vertical", "[0.5,-1,0.5]"), ("h_horizontal", "[0.5,-1,0.5]")]),
    ("Filtros (matriz)", "mediana", filtros.mediana, [("img", "[[10,2,3],[4,50,6],[7,8,9]]"), ("k", "3")]),
    ("Filtros (matriz)", "media_alfa_cortada", filtros.media_alfa_cortada,
     [("img", "[[10,2,3],[4,50,6],[7,8,9]]"), ("k", "3"), ("alpha", "4")]),
    ("Filtros (matriz)", "mediana_ponderada", filtros.mediana_ponderada,
     [("img", "[[10,2,3],[4,50,6],[7,8,9]]"), ("pesos", "[[1,2,1],[2,4,2],[1,2,1]]")]),
    ("Filtros (matriz)", "matrizes_filtro_separavel", filtros.matrizes_filtro_separavel,
     [("shape", "(3,3)"), ("h_vertical", "[0.5,-1,0.5]"), ("h_horizontal", "[0.5,-1,0.5]")]),
    ("Filtros (matriz)", "imagem_integral", filtros.imagem_integral, [("img", F3)]),
    ("Filtros (matriz)", "soma_retangulo", lambda img, i0, j0, i1, j1: filtros.soma_retangulo(
        filtros.imagem_integral(img), i0, j0, i1, j1),
     [("img", F3), ("i0", "0"), ("j0", "0"), ("i1", "1"), ("j1", "1")]),
    ("Filtros (matriz)", "aumentar_resolucao", transformacoes.aumentar_resolucao,
     [("img", "[[10,20],[30,40]]"), ("r", "2")]),
    ("Filtros (matriz)", "diminuir_resolucao", transformacoes.diminuir_resolucao,
     [("img", "[[10,20],[30,40]]"), ("r", "2")]),

    ("Interpolação", "interpolar_bilinear", transformacoes.interpolar_bilinear,
     [("img", "[[10,20,30],[40,50,60],[70,80,90]]"), ("x", "1"), ("y", "1.5")]),
    ("Interpolação", "mapeamento_inverso", lambda s, xl, yl: transformacoes.mapeamento_inverso(
        transformacoes.matriz_similaridade(s), xl, yl),
     [("s", "2"), ("xl", "2"), ("yl", "3")]),
    ("Interpolação", "rbf_interpolacao", transformacoes.rbf_interpolacao,
     [("X", "[0,1,2]"), ("d", "[1,3,2]"), ("xq", "1.8"), ("lam", "0")]),

    ("Energia / MRF", "energia_dados", otimizacao.energia_dados,
     [("f", "[[10,20],[30,40]]"), ("d", "[[12,18],[29,43]]"), ("c", "1")]),
    ("Energia / MRF", "energia_suavidade", otimizacao.energia_suavidade, [("f", "[10,12,15]"), ("ordem", "1")]),
    ("Energia / MRF", "map_por_frequencia", otimizacao.map_por_frequencia,
     [("x", "[1,1,1,0,0,0,0,0,0]"), ("y", "[1,1,0,1,0,0,0,0,1]"), ("y_obs", "0")]),
    ("Energia / MRF", "energia_rotulo_mrf", otimizacao.energia_rotulo_mrf,
     [("rotulo", "0"), ("observado", "0"), ("vizinhos", "[1,1,1,1]"), ("w", "1"), ("s", "0.2")]),

    ("Classificação", "knn_classificar", aprendizado.knn_classificar,
     [("X", "[(1,1),(2,1),(1,2),(4,4),(5,5),(4,5)]"), ("y", "['A','A','A','B','B','B']"),
      ("x", "(3,3)"), ("k", "3")]),
    ("Classificação", "regressao_logistica_prob", aprendizado.regressao_logistica_prob,
     [("w", "[1,-1]"), ("b", "0"), ("x", "[2,1]")]),
    ("Classificação", "svm_vetores_suporte", aprendizado.svm_vetores_suporte,
     [("w", "[1,1]"), ("b", "-3"), ("X", "[(1,4),(2,2),(3,3),(0,0),(0,1),(2,0)]"), ("y", "[1,1,1,-1,-1,-1]")]),
    ("Classificação", "svm_classificar", aprendizado.svm_classificar,
     [("w", "[1,1]"), ("b", "-3"), ("X", "[[2,1.1]]")]),

    ("Agrupamento / PCA", "kmeans", aprendizado.kmeans,
     [("X", "[(1,1),(2,1),(4,4),(5,5)]"), ("k", "None"), ("centroides", "[(4,3),(6,6)]"), ("iteracoes", "2")]),
    ("Agrupamento / PCA", "gmm_passo_e", aprendizado.gmm_passo_e,
     [("X", "[(1,0),(2,1),(3,3),(-1,-1)]"), ("pi", "[0.4,0.6]"), ("mu", "[[0,0],[2,2]]"),
      ("Sigma", "[[[1,0],[0,1]],[[2,0],[0,1]]]")]),
    ("Agrupamento / PCA", "pca", aprendizado.pca, [("X", "[(2,0),(0,2),(3,1),(1,3)]"), ("ddof", "0")]),
]
