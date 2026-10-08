"""Classificação (k-NN, regressão logística, SVM linear), agrupamento (k-means, GMM) e PCA."""
import numpy as np


# ---------- classificadores ----------
def knn_classificar(X, y, x, k=1):
    """k-NN com distância euclidiana. Empates na k-ésima distância entram todos na votação;
    se a votação empatar, desempata pelo voto ponderado por 1/distância.
    Devolve (rótulo, índices dos vizinhos usados)."""
    X, y, x = np.asarray(X, float), np.asarray(y), np.asarray(x, float)
    dist = np.linalg.norm(X - x, axis=1)
    lim = np.sort(dist)[k - 1]
    viz = np.where(dist <= lim + 1e-9)[0]
    classes, cont = np.unique(y[viz], return_counts=True)
    top = classes[cont == cont.max()]
    if len(top) == 1:
        return top[0], viz
    pesos = {c: np.sum(1 / (dist[viz][y[viz] == c] + 1e-12)) for c in top}
    return max(pesos, key=pesos.get), viz


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def regressao_logistica_prob(w, b, x):
    """p(y=1|x) = σ(w·x + b)."""
    return sigmoide(np.asarray(w, float) @ np.asarray(x, float) + b)


def svm_decisao(w, b, X):
    """f(x) = w·x + b."""
    return np.asarray(X, float) @ np.asarray(w, float) + b


def svm_classificar(w, b, X):
    """Classe +1 ou -1 conforme o sinal de f(x)."""
    return np.where(svm_decisao(w, b, X) >= 0, 1, -1)


def svm_vetores_suporte(w, b, X, y, tol=1e-9):
    """Índices dos pontos com menor margem funcional y f(x) (os que estão sobre a margem)."""
    m = np.asarray(y, float) * svm_decisao(w, b, X)
    return np.where(m <= m.min() + tol)[0]


# ---------- k-means ----------
def kmeans(X, k=None, centroides=None, iteracoes=10, seed=0):
    """k-means com distância euclidiana. Devolve (centróides, rótulos)."""
    X = np.asarray(X, float)
    if centroides is None:
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), k, replace=False)].copy()
    else:
        C = np.asarray(centroides, float).copy()
    for _ in range(iteracoes):
        rot = np.linalg.norm(X[:, None] - C[None], axis=2).argmin(1)
        novo = np.array([X[rot == j].mean(0) if np.any(rot == j) else C[j] for j in range(len(C))])
        parar = np.allclose(novo, C)
        C = novo
        if parar:
            break
    rot = np.linalg.norm(X[:, None] - C[None], axis=2).argmin(1)
    return C, rot


def segmentar_imagem_kmeans(img, k=3, iteracoes=10, seed=0):
    """Segmenta a imagem agrupando as cores dos pixels; cada pixel recebe a cor do seu centróide."""
    a = np.asarray(img, float)
    X = a.reshape(-1, 1) if a.ndim == 2 else a.reshape(-1, a.shape[2])
    C, rot = kmeans(X, k=k, iteracoes=iteracoes, seed=seed)
    return C[rot].reshape(a.shape)


# ---------- GMM ----------
def gaussiana_pdf(X, mu, Sigma):
    X, mu, S = np.asarray(X, float), np.asarray(mu, float), np.asarray(Sigma, float)
    d = X - mu
    m = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
    return np.exp(-0.5 * m) / np.sqrt((2 * np.pi) ** len(mu) * np.linalg.det(S))


def gmm_passo_e(X, pi, mu, Sigma):
    """Responsabilidades γ_ik = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)."""
    p = np.stack([pi[k] * gaussiana_pdf(X, mu[k], Sigma[k]) for k in range(len(pi))], axis=1)
    return p / p.sum(axis=1, keepdims=True)


def gmm_passo_m(X, resp, reg=1e-6):
    X = np.asarray(X, float)
    Nk = resp.sum(0)
    pi = Nk / len(X)
    mu = (resp.T @ X) / Nk[:, None]
    Sigma = []
    for k in range(resp.shape[1]):
        d = X - mu[k]
        Sigma.append((resp[:, k, None] * d).T @ d / Nk[k] + reg * np.eye(X.shape[1]))
    return pi, mu, np.array(Sigma)


def gmm_em(X, pi, mu, Sigma, iteracoes=20):
    """EM completo a partir de parâmetros iniciais. Devolve (pi, mu, Sigma, responsabilidades)."""
    X = np.asarray(X, float)
    pi, mu, Sigma = np.asarray(pi, float), np.asarray(mu, float), np.asarray(Sigma, float)
    for _ in range(iteracoes):
        resp = gmm_passo_e(X, pi, mu, Sigma)
        pi, mu, Sigma = gmm_passo_m(X, resp)
    return pi, mu, Sigma, gmm_passo_e(X, pi, mu, Sigma)


# ---------- PCA ----------
def pca(X, ddof=0):
    """Devolve (média, covariância, autovalores desc., autovetores em colunas)."""
    X = np.asarray(X, float)
    media = X.mean(0)
    Xc = X - media
    C = Xc.T @ Xc / (len(X) - ddof)
    vals, vecs = np.linalg.eigh(C)
    ordem = vals.argsort()[::-1]
    return media, C, vals[ordem], vecs[:, ordem]


def variancia_explicada(autovalores):
    v = np.asarray(autovalores, float)
    return v / v.sum()


def projetar_pca(X, media, autovetores, n):
    return (np.asarray(X, float) - media) @ autovetores[:, :n]
