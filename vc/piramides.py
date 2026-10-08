"""Pirâmides Gaussiana e Laplaciana e blending de imagens (Burt e Adelson, 1983).

As imagens são arrays NumPy (H, W) ou (H, W, C); cada canal é tratado separadamente.
"""
import numpy as np

from . import filtros

# Núcleo gaussiano 5x5 de Burt-Adelson (binomial [1 4 6 4 1] / 16).
NUCLEO = filtros.nucleo_binomial(5)


def reduzir(img):
    """REDUCE: suaviza com o núcleo gaussiano e mantém 1 amostra a cada 2 em cada eixo."""
    return filtros.convolucao(img, NUCLEO, "reflect")[::2, ::2]


def expandir(img, shape):
    """EXPAND: insere zeros entre as amostras, suaviza (x4 para repor a energia) e recorta para `shape` (H, W)."""
    a = np.asarray(img, float)
    z = np.zeros((a.shape[0] * 2, a.shape[1] * 2) + a.shape[2:])
    z[::2, ::2] = a
    return (4 * filtros.convolucao(z, NUCLEO, "reflect"))[:shape[0], :shape[1]]


def piramide_gaussiana(img, niveis):
    """Lista [G0, G1, ..., G_{niveis-1}], G0 = img e G_{i+1} = REDUCE(G_i)."""
    g = [np.asarray(img, float)]
    for _ in range(niveis - 1):
        g.append(reduzir(g[-1]))
    return g


def piramide_laplaciana(img, niveis):
    """Lista [L0, ..., L_{n-1}] com L_i = G_i - EXPAND(G_{i+1}) e L_{n-1} = G_{n-1} (resíduo passa-baixas)."""
    g = piramide_gaussiana(img, niveis)
    return [g[i] - expandir(g[i + 1], g[i].shape[:2]) for i in range(niveis - 1)] + [g[-1]]


def colapsar(lap):
    """Reconstrói a imagem a partir da pirâmide Laplaciana: G_i = L_i + EXPAND(G_{i+1})."""
    r = lap[-1]
    for L in reversed(lap[:-1]):
        r = L + expandir(r, L.shape[:2])
    return r


def blending(a, b, mascara, niveis=5):
    """Mistura `a` (onde mascara = 1) e `b` (onde mascara = 0) em cada banda de frequência.

    L_blend_i = Gm_i * La_i + (1 - Gm_i) * Lb_i, em que Gm é a pirâmide Gaussiana da máscara:
    nos níveis de baixa frequência a transição é larga e suave; nos de alta, estreita (detalhes preservados).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.asarray(mascara, float)
    if m.max() > 1:
        m = m / 255.0
    if m.ndim == 3:
        m = m.mean(axis=2)
    if a.shape != b.shape or m.shape != a.shape[:2]:
        raise ValueError("Imagens e máscara devem ter o mesmo tamanho.")
    if a.ndim == 3:
        m = m[..., None]
    gm = piramide_gaussiana(m, niveis)
    la, lb = piramide_laplaciana(a, niveis), piramide_laplaciana(b, niveis)
    return colapsar([w * x + (1 - w) * y for w, x, y in zip(gm, la, lb)])


def justaposicao(a, b, mascara):
    """Colagem direta (sem mistura): pixels de `a` onde a máscara é 1 e de `b` onde é 0."""
    m = np.asarray(mascara, float)
    if m.max() > 1:
        m = m / 255.0
    if np.asarray(a).ndim == 3 and m.ndim == 2:
        m = m[..., None]
    return m * np.asarray(a, float) + (1 - m) * np.asarray(b, float)
