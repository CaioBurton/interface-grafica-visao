"""Modelos de iluminação e óptica."""
import numpy as np


def phong_especular(Li, ks, theta_s_graus, ke):
    """Intensidade especular: Ls = ks Li cos^ke(θs)."""
    return ks * Li * max(0.0, np.cos(np.deg2rad(theta_s_graus))) ** ke


def reflexao_difusa(Li, kd, n, v):
    """Intensidade difusa: Ld = kd Li [v̂ · n̂]+ (v = direção da luz, apontando para a luz)."""
    n, v = np.asarray(n, float), np.asarray(v, float)
    c = (n / np.linalg.norm(n)) @ (v / np.linalg.norm(v))
    return kd * Li * max(0.0, c)


def lente_fina_distancia_focal(z0, zi):
    """1/z0 + 1/zi = 1/f."""
    return 1.0 / (1.0 / z0 + 1.0 / zi)


def numero_f(f, d):
    """Número f: N = f / d."""
    return f / d
