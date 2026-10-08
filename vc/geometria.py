"""Geometria 2D/3D em coordenadas homogêneas e rotações."""
import numpy as np


def para_homogeneo(p):
    return np.append(np.asarray(p, float), 1.0)


def de_homogeneo(p):
    p = np.asarray(p, float)
    if np.isclose(p[-1], 0):
        raise ValueError("Ponto no infinito (última coordenada homogênea é zero).")
    return p[:-1] / p[-1]


def reta_por_pontos(p, q):
    """Reta (co-vetor) que passa pelos pontos 2D p e q."""
    return np.cross(para_homogeneo(p), para_homogeneo(q))


def intersecao_retas(a, b):
    """Ponto (x, y) de interseção de duas retas a~ e b~ (produto vetorial)."""
    return de_homogeneo(np.cross(np.asarray(a, float), np.asarray(b, float)))


def aplicar_afim(A, p):
    """x' = A [x y 1]^T, com A de tamanho 2x3 (2D) ou 3x4 (3D)."""
    return np.asarray(A, float) @ para_homogeneo(p)


def aplicar_projetiva(H, p):
    """Transformação projetiva (homografia) de um ponto: x~' = H x~, normalizado."""
    return de_homogeneo(np.asarray(H, float) @ para_homogeneo(p))


def transformar_reta(l, H):
    """Co-vetor da reta transformado por H: l~' = H^-T l~."""
    return np.linalg.inv(np.asarray(H, float)).T @ np.asarray(l, float)


def quaternion_eixo_angulo(n, theta_graus):
    """Quatérnio unitário (w, x, y, z) para rotação de theta em torno do eixo n."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    t = np.deg2rad(theta_graus) / 2
    return np.array([np.cos(t), *(n * np.sin(t))])


def quaternion_para_matriz(q):
    """Matriz de rotação 3x3 a partir de um quatérnio (w, x, y, z)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matriz_rotacao_eixo_angulo(n, theta_graus):
    """Fórmula de Rodrigues: R = I + sen(θ)[n]x + (1 - cos θ)[n]x²."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    t = np.deg2rad(theta_graus)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
