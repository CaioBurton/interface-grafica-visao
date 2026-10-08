"""Modelos de projeção e câmera."""
import numpy as np


def projecao_ortografica(p, s=1.0):
    """Projeção ortográfica escalada: x' = s [I2 | 0] p (aceita (3,) ou (N,3))."""
    return s * np.asarray(p, float)[..., :2]


def projecao_perspectiva(p, f=1.0):
    """Projeção perspectiva: (f x/z, f y/z)."""
    p = np.asarray(p, float)
    return f * p[..., :2] / p[..., 2:3]


def distancia_focal_fov(largura_sensor, fov_graus):
    """f = (W/2) / tan(θ/2)."""
    return (largura_sensor / 2) / np.tan(np.deg2rad(fov_graus) / 2)


def distorcao_radial(xy, k1, k2=0.0):
    """Aplica x̂ = x (1 + k1 r² + k2 r⁴) em coordenadas normalizadas da câmera."""
    xy = np.asarray(xy, float)
    r2 = np.sum(xy ** 2, axis=-1, keepdims=True)
    return xy * (1 + k1 * r2 + k2 * r2 ** 2)


def para_pixel(xy, f, cx, cy):
    """Coordenadas normalizadas -> pixel: u = f x + cx, v = f y + cy."""
    xy = np.asarray(xy, float)
    return np.stack([f * xy[..., 0] + cx, f * xy[..., 1] + cy], axis=-1)
