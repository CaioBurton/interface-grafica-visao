"""Formação geométrica da imagem: cubo 3D, parâmetros extrínsecos/intrínsecos e projeções.

Todas as transformações e projeções são implementadas aqui com NumPy (sem funções prontas
de computação gráfica/visão computacional).
"""
import numpy as np

RESOLUCOES = {"640 x 480": (640, 480), "800 x 600": (800, 600), "1280 x 720": (1280, 720)}

# Vértices P1..P8 (linhas 0..7) no sistema de coordenadas do mundo.
VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)

# Arestas (índices 0-based): P1–P2, P2–P3, ..., P4–P8.
ARESTAS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)]

PADRAO_PERSPECTIVA = {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "skew": 0.0}
PADRAO_ESCALA = 190.0
PADRAO_EXTRINSECOS = {"tx": 0.0, "ty": 0.0, "tz": 5.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}

Z_MIN = 1e-3  # plano de recorte próximo (perspectiva): z da câmera deve ser maior que isto


def rotacao_x(graus):
    c, s = np.cos(np.deg2rad(graus)), np.sin(np.deg2rad(graus))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotacao_y(graus):
    c, s = np.cos(np.deg2rad(graus)), np.sin(np.deg2rad(graus))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotacao_z(graus):
    c, s = np.cos(np.deg2rad(graus)), np.sin(np.deg2rad(graus))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matriz_rotacao(rx, ry, rz):
    """R = Rz Ry Rx (ângulos em graus)."""
    return rotacao_z(rz) @ rotacao_y(ry) @ rotacao_x(rx)


def matriz_extrinseca(tx, ty, tz, rx, ry, rz):
    """Matriz [R | t] (3x4): mundo -> câmera, Xc = R Xw + t."""
    return np.hstack([matriz_rotacao(rx, ry, rz), np.array([[tx], [ty], [tz]], float)])


def matriz_intrinseca(fx, fy, cx, cy, skew=0.0):
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def mundo_para_camera(pontos, extrinseca):
    """Aplica [R | t] a pontos (N,3) em coordenadas do mundo."""
    pontos = np.asarray(pontos, float)
    return pontos @ extrinseca[:, :3].T + extrinseca[:, 3]


def projetar_perspectiva(pc, K):
    """Pontos da câmera (N,3) -> pixels (N,2): x~ = K Xc, (u,v) = (x~1/x~3, x~2/x~3).

    Pontos com z <= 0 resultam em NaN.
    """
    pc = np.asarray(pc, float)
    h = pc @ K.T
    uv = np.full((len(pc), 2), np.nan)
    ok = pc[:, 2] > 0
    uv[ok] = h[ok, :2] / h[ok, 2:3]
    return uv


def projetar_ortografica_escalada(pc, escala, cx, cy):
    """Pontos da câmera (N,3) -> pixels (N,2): u = s x + cx, v = s y + cy (z é descartado)."""
    pc = np.asarray(pc, float)
    return np.stack([escala * pc[:, 0] + cx, escala * pc[:, 1] + cy], axis=-1)


def _recortar_aresta(a, b):
    """Recorta o segmento ab (coordenadas da câmera) em z >= Z_MIN. Retorna (a, b) ou None."""
    if a[2] < Z_MIN and b[2] < Z_MIN:
        return None
    if a[2] < Z_MIN:
        a = a + (Z_MIN - a[2]) / (b[2] - a[2]) * (b - a)
    elif b[2] < Z_MIN:
        b = b + (Z_MIN - b[2]) / (a[2] - b[2]) * (a - b)
    return a, b


def projetar_cubo(modelo, resolucao, extrinsecos, intrinsecos=None, escala=PADRAO_ESCALA):
    """Projeta o cubo e devolve (vertices_2d (8,2), segmentos [(p, q), ...]).

    modelo: "perspectiva" ou "ortografica". extrinsecos: dict com tx,ty,tz,rx,ry,rz.
    intrinsecos: dict fx,fy,cx,cy,skew (perspectiva). Na ortográfica, o centro da imagem
    é o ponto principal. Vértices atrás da câmera (perspectiva) vêm como NaN.
    """
    E = matriz_extrinseca(*(extrinsecos[k] for k in ("tx", "ty", "tz", "rx", "ry", "rz")))
    pc = mundo_para_camera(VERTICES, E)
    segmentos = []
    if modelo == "perspectiva":
        K = matriz_intrinseca(*(intrinsecos[k] for k in ("fx", "fy", "cx", "cy", "skew")))
        v2d = projetar_perspectiva(pc, K)
        for i, j in ARESTAS:
            r = _recortar_aresta(pc[i], pc[j])
            if r is not None:
                p, q = projetar_perspectiva(np.array(r), K)
                segmentos.append((p, q))
    else:
        cx, cy = resolucao[0] / 2, resolucao[1] / 2
        v2d = projetar_ortografica_escalada(pc, escala, cx, cy)
        segmentos = [(v2d[i], v2d[j]) for i, j in ARESTAS]
    return v2d, segmentos


def renderizar(v2d, segmentos, resolucao):
    """Desenha vértices e arestas numa imagem PIL (fundo escuro)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", resolucao, (20, 20, 24))
    d = ImageDraw.Draw(img)
    for p, q in segmentos:
        d.line([(p[0], p[1]), (q[0], q[1])], fill=(90, 170, 255), width=2)
    for k, (u, v) in enumerate(v2d, start=1):
        if np.isfinite(u) and np.isfinite(v):
            d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(255, 190, 60))
            d.text((u + 6, v - 12), f"P{k}", fill=(230, 230, 230))
    return img
