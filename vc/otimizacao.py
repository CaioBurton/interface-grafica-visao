"""Energias de dados/suavidade, MAP e MRF binário."""
import numpy as np


def energia_dados(f, d, c=1.0):
    """E_D = Σ c (f - d)²."""
    return float(np.sum(np.asarray(c, float) * (np.asarray(f, float) - np.asarray(d, float)) ** 2))


def energia_suavidade(f, ordem=1):
    """Σ (diferença de ordem `ordem`)² ao longo de cada eixo (ordem 1: (f(i+1) - f(i))²)."""
    f = np.asarray(f, float)
    return float(sum(np.sum(np.diff(f, n=ordem, axis=a) ** 2) for a in range(f.ndim)))


def estimar_probabilidades(x, y):
    """Prior p(x) e verossimilhança p(y|x) estimadas pela frequência (rótulos inteiros 0..n)."""
    x, y = np.asarray(x, int), np.asarray(y, int)
    nx, ny = x.max() + 1, y.max() + 1
    prior = np.bincount(x, minlength=nx) / x.size
    veross = np.zeros((nx, ny))
    for xi in range(nx):
        sel = y[x == xi]
        if sel.size:
            veross[xi] = np.bincount(sel, minlength=ny) / sel.size
    return prior, veross


def rotulo_map(y_obs, prior, veross):
    """Rótulo MAP: argmax_x p(y|x) p(x). Devolve (rótulo, posteriores não normalizadas)."""
    post = np.asarray(veross)[:, int(y_obs)] * np.asarray(prior)
    return int(post.argmax()), post


def map_por_frequencia(x, y, y_obs):
    prior, veross = estimar_probabilidades(x, y)
    return rotulo_map(y_obs, prior, veross)


def energia_rotulo_mrf(rotulo, observado, vizinhos, w=1.0, s=0.2):
    """E(f) = (f - d)² + s Σ_{vizinhos} w [f ≠ f_viz]."""
    return (rotulo - observado) ** 2 + s * w * sum(rotulo != v for v in vizinhos)


def restaurar_icm(observado, s=0.2, w=1.0, iteracoes=10):
    """Restauração de imagem binária por ICM (vizinhança N4, atualização em xadrez)."""
    d = (np.asarray(observado) > 0).astype(float)
    f = d.copy()
    ii, jj = np.indices(d.shape)
    for _ in range(iteracoes):
        mudou = False
        for paridade in (0, 1):
            p = np.pad(f, 1)
            n1 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            vz = np.pad(np.ones_like(f), 1)
            n = vz[:-2, 1:-1] + vz[2:, 1:-1] + vz[1:-1, :-2] + vz[1:-1, 2:]
            E0 = d ** 2 + s * w * n1
            E1 = (1 - d) ** 2 + s * w * (n - n1)
            novo = (E1 < E0).astype(float)
            sel = (ii + jj) % 2 == paridade
            mudou |= bool(np.any(f[sel] != novo[sel]))
            f[sel] = novo[sel]
        if not mudou:
            break
    return f
