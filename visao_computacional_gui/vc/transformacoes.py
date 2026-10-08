"""Interpolação, transformações geométricas, reamostragem, composição e RBF."""
import numpy as np

from . import camera, filtros


# ---------- interpolação ----------
def interpolar_vizinho(img, x, y, fora=0.0):
    """Vizinho mais próximo; x = coluna, y = linha."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    out = img[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
    if img.ndim == 3:
        ok = ok[..., None]
    return np.where(ok, out, fora)


def interpolar_bilinear(img, x, y, fora=0.0):
    """Interpolação bilinear em (x, y) com x = coluna e y = linha (aceita arrays).
    Pontos fora da imagem recebem `fora`."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    eps = 1e-9
    ok = (x >= -eps) & (x <= W - 1 + eps) & (y >= -eps) & (y <= H - 1 + eps)
    xa, xb = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
    ya, yb = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    if img.ndim == 3:
        dx, dy, ok = dx[..., None], dy[..., None], ok[..., None]
    out = ((1 - dx) * (1 - dy) * img[ya, xa] + dx * (1 - dy) * img[ya, xb]
           + (1 - dx) * dy * img[yb, xa] + dx * dy * img[yb, xb])
    return np.where(ok, out, fora)


_INTERPOLADORES = {"bilinear": interpolar_bilinear, "vizinho": interpolar_vizinho}


# ---------- transformações geométricas ----------
def matriz_similaridade(s=1.0, theta_graus=0.0, t=(0.0, 0.0)):
    """Matriz 3x3 homogênea x' = s R x + t."""
    th = np.deg2rad(theta_graus)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.eye(3)
    M[:2, :2] = s * R
    M[:2, 2] = t
    return M


def similaridade_centrada(shape, s=1.0, theta_graus=0.0, t=(0.0, 0.0)):
    """Similaridade em torno do centro da imagem (escala/rotação), seguida de translação t."""
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    T = np.eye(3)
    T[:2, 2] = (cx, cy)
    T0 = np.eye(3)
    T0[:2, 2] = (-cx, -cy)
    return matriz_similaridade(1, 0, t) @ T @ matriz_similaridade(s, theta_graus) @ T0


def mapeamento_inverso(M, xl, yl):
    """Dado (x', y') na saída, devolve (x, y) na entrada: x = M^-1 x' (com divisão homogênea)."""
    M = np.asarray(M, float)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]])
    xl, yl = np.asarray(xl, float), np.asarray(yl, float)
    p = np.linalg.inv(M) @ np.stack([xl.ravel(), yl.ravel(), np.ones(xl.size)])
    return (p[0] / p[2]).reshape(xl.shape), (p[1] / p[2]).reshape(yl.shape)


def transformar_imagem(img, M, saida_shape=None, interpolacao="bilinear"):
    """Aplica a transformação M (entrada -> saída) por mapeamento inverso + interpolação."""
    img = np.asarray(img, float)
    H, W = saida_shape or img.shape[:2]
    yl, xl = np.mgrid[0:H, 0:W]
    x, y = mapeamento_inverso(M, xl, yl)
    return _INTERPOLADORES[interpolacao](img, x, y)


# ---------- reamostragem ----------
def nucleo_linear(r):
    """Triângulo de interpolação linear para fator r (r=2 -> [0.5, 1, 0.5])."""
    k = np.arange(-(r - 1), r)
    return 1 - np.abs(k) / r


def aumentar_resolucao(img, r=2):
    """Upsampling: insere zeros (r-1 entre amostras) e filtra com o núcleo linear."""
    a = np.asarray(img, float)
    z = np.zeros((a.shape[0] * r, a.shape[1] * r) + a.shape[2:])
    z[::r, ::r] = a
    t = nucleo_linear(r)
    return filtros.convolucao(z, np.outer(t, t))


def diminuir_resolucao(img, r=2):
    """Downsampling: filtra (pré-filtro linear normalizado, zero-padding) e mantém 1 a cada r amostras."""
    t = nucleo_linear(r) / r
    return filtros.convolucao(img, np.outer(t, t))[::r, ::r]


# ---------- composição ----------
def composicao_over(frente, fundo, alpha):
    """Operador Over: C = (1 - α) B + α F (α escalar ou mapa (H, W))."""
    F, B, a = np.asarray(frente, float), np.asarray(fundo, float), np.asarray(alpha, float)
    if a.ndim == 2 and F.ndim == 3:
        a = a[..., None]
    return (1 - a) * B + a * F


# ---------- distorção radial em imagens ----------
def corrigir_distorcao_radial(img, k1, k2=0.0, f=None):
    """Para cada pixel da saída (sem distorção) busca na entrada o ponto distorcido correspondente."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    f = f or max(H, W)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    v, u = np.mgrid[0:H, 0:W]
    xy = np.stack([(u - cx) / f, (v - cy) / f], axis=-1)
    px = camera.para_pixel(camera.distorcao_radial(xy, k1, k2), f, cx, cy)
    return interpolar_bilinear(img, px[..., 0], px[..., 1])


# ---------- RBF ----------
def distancia_euclidiana(r):
    return r


def _pontos(X):
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


def rbf_pesos(X, d, phi=distancia_euclidiana, lam=0.0):
    """Resolve (Φ + λI) w = d com Φij = φ(||xi - xj||)."""
    X = _pontos(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return np.linalg.solve(phi(D) + lam * np.eye(len(X)), np.asarray(d, float))


def rbf_interpolar(xq, X, w, phi=distancia_euclidiana):
    """f(x) = Σ wk φ(||x - xk||). xq: escalar/(m,) para dados 1D, ou (dim,)/(m, dim) para dados nD."""
    X = _pontos(X)
    Q = np.asarray(xq, float)
    if X.shape[1] == 1:
        escalar = Q.ndim == 0
        Q = Q.reshape(-1, 1)
    else:
        escalar = Q.ndim == 1
        Q = np.atleast_2d(Q)
    D = np.linalg.norm(Q[:, None] - X[None], axis=2)
    v = phi(D) @ np.asarray(w, float)
    return v[0] if escalar else v


def rbf_interpolacao(X, d, xq, lam=0.0):
    """Conveniência: devolve (pesos, valores interpolados em xq) com φ = distância euclidiana."""
    w = rbf_pesos(X, d, lam=lam)
    return w, rbf_interpolar(xq, X, w)
