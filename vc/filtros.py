"""Filtragem linear e não linear, forma matricial e imagem integral.

As imagens são arrays NumPy (H, W) ou (H, W, C); cada canal é filtrado separadamente.
`modo` é o modo de preenchimento de borda do np.pad ("constant" = zero-padding).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _por_canal(img, fn):
    a = np.asarray(img, float)
    if a.ndim == 2:
        return fn(a)
    return np.stack([fn(a[..., c]) for c in range(a.shape[2])], axis=-1)


def _janelas(canal, kh, kw, modo):
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("A janela deve ter dimensões ímpares.")
    p = np.pad(canal, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode=modo)
    return sliding_window_view(p, (kh, kw))


# ---------- núcleos ----------
def nucleo_media(k):
    return np.ones((k, k)) / (k * k)


def nucleo_gaussiano(k, sigma=None):
    sigma = sigma or k / 4
    x = np.arange(k) - k // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def nucleo_binomial(k):
    """Pesos binomiais (linha do triângulo de Pascal) 2D normalizados."""
    b = np.array([1.0])
    for _ in range(k - 1):
        b = np.convolve(b, [1, 1])
    b /= b.sum()
    return np.outer(b, b)


H_CORNER = np.array([1, -2, 1]) / 2


def nucleo_corner():
    """Filtro corner = 1/4 [1 -2 1; -2 4 -2; 1 -2 1] (separável: h^T h com h = 1/2[1 -2 1])."""
    return np.outer(H_CORNER, H_CORNER)


# ---------- convolução ----------
def convolucao(img, nucleo, modo="constant"):
    """Convolução 2D (núcleo espelhado) com preenchimento de borda."""
    k = np.asarray(nucleo, float)[::-1, ::-1]
    return _por_canal(img, lambda c: np.einsum("ijkl,kl->ij", _janelas(c, *k.shape, modo), k))


def filtro_separavel(img, h_vertical, h_horizontal, modo="constant"):
    """Aplica o filtro 1D horizontal e depois o vertical (g = Hv Hh f)."""
    h = np.asarray(h_horizontal, float)[None, :]
    v = np.asarray(h_vertical, float)[:, None]
    return convolucao(convolucao(img, h, modo), v, modo)


# ---------- filtros não lineares ----------
def mediana(img, k=3, modo="constant"):
    return _por_canal(img, lambda c: np.median(_janelas(c, k, k, modo).reshape(*c.shape, -1), axis=-1))


def media_alfa_cortada(img, k=3, alpha=2, modo="constant"):
    """Ordena a janela, descarta alpha//2 menores e alpha//2 maiores valores e tira a média."""
    t = int(alpha) // 2

    def f(c):
        w = np.sort(_janelas(c, k, k, modo).reshape(*c.shape, -1), axis=-1)
        if w.shape[-1] - 2 * t < 1:
            raise ValueError("alpha grande demais para a janela.")
        return w[..., t:w.shape[-1] - t].mean(axis=-1)
    return _por_canal(img, f)


def mediana_ponderada(img, pesos, modo="constant"):
    """Mediana em que cada valor da janela conta `pesos` vezes (valor onde o peso acumulado atinge metade)."""
    pesos = np.asarray(pesos, float)

    def f(c):
        win = _janelas(c, *pesos.shape, modo).reshape(*c.shape, -1)
        idx = np.argsort(win, axis=-1, kind="stable")
        vs = np.take_along_axis(win, idx, -1)
        cum = np.cumsum(pesos.ravel()[idx], axis=-1)
        pos = (cum >= cum[..., -1:] / 2).argmax(-1)
        return np.take_along_axis(vs, pos[..., None], -1)[..., 0]
    return _por_canal(img, f)


# ---------- forma matricial ----------
def matriz_filtro_1d(shape, h, eixo):
    """Matriz (HW x HW) que aplica o filtro 1D `h` (ímpar) na imagem vetorizada linha a linha.
    eixo=1: horizontal (H_h); eixo=0: vertical (H_v). Zero-padding nas bordas."""
    H, W = shape
    h = np.asarray(h, float)
    r = len(h) // 2
    M = np.zeros((H * W, H * W))
    for i in range(H):
        for j in range(W):
            for d in range(-r, r + 1):
                ii, jj = (i, j + d) if eixo == 1 else (i + d, j)
                if 0 <= ii < H and 0 <= jj < W:
                    M[i * W + j, ii * W + jj] = h[d + r]
    return M


def matrizes_filtro_separavel(shape, h_vertical, h_horizontal):
    """Retorna (H_v, H_h) tais que g = H_v @ H_h @ f.ravel()."""
    return matriz_filtro_1d(shape, h_vertical, 0), matriz_filtro_1d(shape, h_horizontal, 1)


def aplicar_matricial(img, Hv, Hh):
    f = np.asarray(img, float)
    return (Hv @ (Hh @ f.ravel())).reshape(f.shape)


# ---------- imagem integral ----------
def imagem_integral(img):
    """s(i,j) = soma de f(k,l) para k<=i, l<=j."""
    return np.asarray(img, float).cumsum(axis=0).cumsum(axis=1)


def soma_retangulo(s, i0, j0, i1, j1):
    """Soma dos pixels do retângulo [i0..i1] x [j0..j1] usando a imagem integral s."""
    total = s[i1, j1]
    if i0 > 0:
        total = total - s[i0 - 1, j1]
    if j0 > 0:
        total = total - s[i1, j0 - 1]
    if i0 > 0 and j0 > 0:
        total = total + s[i0 - 1, j0 - 1]
    return total


def filtro_media_integral(img, k=3):
    """Filtro de média k x k (zero-padding) calculado em tempo constante por pixel via imagem integral."""
    r = k // 2

    def f(c):
        p = np.pad(c, r)
        s = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
        s[1:, 1:] = p.cumsum(0).cumsum(1)
        return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)
    return _por_canal(img, f)
