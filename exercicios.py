"""Resolução das tarefas práticas da 1ª unidade (cada função devolve o passo a passo em texto)."""
import numpy as np


def M(a):
    """Formata matriz/vetor para exibição."""
    return np.array2string(np.asarray(a, dtype=float), precision=4, suppress_small=True, floatmode="maxprec")


def L(*linhas):
    return "\n".join(str(x) for x in linhas)


# ======================= Slide 32 =======================
def s32_1():
    a, b = np.array([2, -1, 3.0]), np.array([1, 1, -5.0])
    p = np.cross(a, b)
    return L("p~ = a~ x b~ (produto vetorial em coordenadas homogêneas)",
             f"p~ = {M(p)}", f"p = (x, y) = (p1/p3, p2/p3) = ({p[0]/p[2]:.4f}, {p[1]/p[2]:.4f})")


def s32_2():
    A = np.array([[2, 0, 1], [0, 3, -1.0]])
    p = np.array([2, 1, 1.0])
    r = A @ p
    return L("x' = A * [x y 1]^T", f"A =\n{M(A)}", f"x' = {M(r)}  ->  p' = ({r[0]:g}, {r[1]:g})")


def s32_3():
    l = np.array([2, -3, 5.0])
    H = np.array([[1, 0.5, 2], [0, 2, -1], [0, 0.2, 1]])
    lp = np.linalg.inv(H).T @ l
    return L("Retas se transformam com a inversa transposta: l~' = H^-T * l~",
             f"H^-T =\n{M(np.linalg.inv(H).T)}", f"l~' = {M(lp)}",
             f"(normalizado por l3: {M(lp / lp[2])})")


def s32_4():
    n = np.array([0.6, 0.8, 0.0])
    th = np.deg2rad(60)
    w, v = np.cos(th / 2), n * np.sin(th / 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return L("q = (cos(θ/2), n sin(θ/2))",
             f"q = (w, x, y, z) = ({w:.4f}, {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})   |q| = {np.linalg.norm([w, *v]):.4f}",
             "Matriz de rotação (Rodrigues): R = I + sinθ [n]x + (1 - cosθ) [n]x²", f"R =\n{M(R)}")


# ======================= Slide 36 =======================
def s36_1():
    p = np.array([3, -2, 5.0])
    s = 2
    return L("x' = s [I2 | 0] p", f"x' = ({s*p[0]:g}, {s*p[1]:g})")


def s36_2():
    p = np.array([4, 6, 2.0])
    return L("Projeção perspectiva (f = 1): x̄ = (x/z, y/z)", f"x̄ = ({p[0]/p[2]:g}, {p[1]/p[2]:g})")


def s36_3():
    W, th = 36, np.deg2rad(60)
    f = (W / 2) / np.tan(th / 2)
    return L("tan(θH/2) = (W/2)/f  ->  f = (W/2) / tan(θH/2)", f"f = 18 / tan(30°) = {f:.4f} mm")


def s36_4():
    x, y, k1, k2 = 0.5, 0.3, 0.10, 0.05
    f, cx, cy = 800, 320, 240
    r2 = x * x + y * y
    fator = 1 + k1 * r2 + k2 * r2 ** 2
    xh, yh = x * fator, y * fator
    u, v = f * xh + cx, f * yh + cy
    return L(f"r² = {r2:.4f}", f"fator = 1 + k1 r² + k2 r⁴ = {fator:.5f}",
             f"(x̂, ŷ) = ({xh:.5f}, {yh:.5f})", "Pixel: u = f x̂ + cx ; v = f ŷ + cy",
             f"(u, v) = ({u:.2f}, {v:.2f})")


# ======================= Slide 40 =======================
def s40_1():
    Li, ks, th, ke = 200, 0.5, np.deg2rad(60), 10
    r = ks * Li * np.cos(th) ** ke
    return L("Ls = ks Li cos^ke(θs)", f"Ls = 0.5 * 200 * cos(60°)^10 = {r:.5f}")


def s40_2():
    n = np.array([0, 0, 1.0])
    v = np.array([np.sqrt(2) / 2, 0, np.sqrt(2) / 2])
    Li, kd = 200, 0.8
    c = max(0.0, n @ v)
    return L("Ld = kd Li [v̂i · n̂]+", f"v̂i · n̂ = {c:.5f}", f"Ld = 0.8 * 200 * {c:.5f} = {kd*Li*c:.4f}")


def s40_3():
    z0, zi = 50, 25
    f = 1 / (1 / z0 + 1 / zi)
    return L("1/z0 + 1/zi = 1/f", f"1/f = 1/50 + 1/25 = 3/50  ->  f = {f:.4f} cm")


def s40_4():
    f, d = 50, 25
    return L("N = f / d", f"N = 50 / 25 = {f/d:g}  (f/{f/d:g})")


# ======================= Slide 42 =======================
def s42_1():
    B, F, a = 100, 200, 0.3
    return L("Over: C = (1 - α) B + α F", f"C = 0.7*100 + 0.3*200 = {(1-a)*B + a*F:g}")


def _banda(n, tam=3):
    """Matriz esparsa (densa aqui) do filtro 1D 1/2[1 -2 1] aplicado em 'tam' x 'tam' (row-major)."""
    h = np.array([1, -2, 1]) / 2
    N = tam * tam
    Hh, Hv = np.zeros((N, N)), np.zeros((N, N))
    for i in range(tam):
        for j in range(tam):
            k = i * tam + j
            for d in (-1, 0, 1):
                if 0 <= j + d < tam:
                    Hh[k, i * tam + j + d] = h[d + 1]
                if 0 <= i + d < tam:
                    Hv[k, (i + d) * tam + j] = h[d + 1]
    return Hh, Hv


def s42_2():
    f = np.arange(1, 10, dtype=float).reshape(3, 3)
    h = np.array([1, -2, 1]) / 2
    # a) filtro corner separável: 1/2[1 -2 1]^T * 1/2[1 -2 1]  -> aqui aplicado como h (horizontal) e depois h (vertical)
    horiz = (f @ h).reshape(-1, 1)   # h aplicado à linha inteira (centrado na coluna central)
    vert = (horiz[:, 0] * h).sum()
    # b)
    Hh, Hv = _banda(3)
    fv = f.reshape(-1, 1)
    g = Hv @ (Hh @ fv)
    # c)
    s = f.cumsum(0).cumsum(1)
    i0, j0, i1, j1 = 0, 0, 1, 1
    soma = s[i1, j1] - (s[i0 - 1, j1] if i0 > 0 else 0) - (s[i1, j0 - 1] if j0 > 0 else 0) \
        + (s[i0 - 1, j0 - 1] if i0 > 0 and j0 > 0 else 0)
    fmt = lambda A: "\n".join(" ".join(f"{x:5.1f}" for x in r) for r in A)
    return L("f =\n" + M(f),
             "",
             "a) Corner separável: horizontal h = 1/2[1 -2 1] em cada linha, depois vertical na coluna do pixel central.",
             f"   Após horizontal, coluna central = {M(horiz[:, 0])}",
             f"   Saída do pixel central = {vert:g}",
             "",
             "b) g = Hv (Hh f), com f em vetor (linha a linha) e zero-padding nas bordas:",
             f"   f^T = {M(fv.ravel())}",
             "   Hh (9x9):\n" + fmt(Hh),
             "   Hv (9x9):\n" + fmt(Hv),
             f"   g^T = {M(g.ravel())}  ->  pixel central g[4] = {g[4,0]:g}",
             "",
             "c) Imagem integral s(i,j) = Σ f(k,l), k≤i, l≤j:",
             M(s),
             f"   Soma do retângulo (0,0)-(1,1) = s(1,1) = {soma:g}   (confere: 1+2+4+5 = 12)")


# ======================= Slide 33 =======================
def _mediana_ponderada(v, w):
    idx = np.argsort(v, kind="stable")
    v, w = np.asarray(v)[idx], np.asarray(w)[idx]
    cum = np.cumsum(w)
    return v[np.searchsorted(cum, cum[-1] / 2)], v, w, cum


def s33_1():
    f = np.array([[10, 2, 3], [4, 50, 6], [7, 8, 9.0]])
    W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1.0]])
    v = np.sort(f.ravel())
    med = np.median(v)
    alpha = 4
    cortada = v[alpha // 2: len(v) - alpha // 2]
    mp, vs, ws, cum = _mediana_ponderada(f.ravel(), W.ravel())
    return L(f"Valores ordenados: {M(v)}",
             f"a) Mediana = {med:g}",
             f"b) Média α-cortada (α = {alpha}: descarta os {alpha//2} menores e os {alpha//2} maiores): "
             f"{M(cortada)} -> média = {cortada.mean():.4f}",
             f"   (se α fosse descartado em cada extremo, restaria só a mediana {med:g})",
             f"c) Mediana ponderada (cada valor repetido w vezes): valores {M(vs)}",
             f"   pesos {M(ws)}, acumulado {M(cum)} (total {cum[-1]:g}, metade {cum[-1]/2:g})",
             f"   Mediana ponderada = {mp:g}")


def _conv_same(f, k):
    p = k.shape[0] // 2
    fp = np.pad(f.astype(float), p)
    out = np.zeros(f.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i, j] = (fp[i:i + k.shape[0], j:j + k.shape[1]] * k).sum()
    return out


def s33_2():
    f = np.array([[10, 20], [30, 40.0]])
    r = 2
    # a) upsampling
    z = np.zeros((4, 4))
    z[::r, ::r] = f
    hu = np.array([0.5, 1, 0.5])
    gu = _conv_same(z, np.outer(hu, hu))
    # b) downsampling
    hd = np.array([1, 2, 1]) / 4
    fil = _conv_same(f, np.outer(hd, hd))
    gd = fil[::r, ::r]
    return L("a) Upsampling r=2, filtro linear 1/2[1 2 1] (separável)",
             "   Imagem após inserção de zeros:\n" + M(z),
             "   Após o filtro:\n" + M(gu),
             f"   g(1,1) = {gu[1,1]:g}   (média dos 4 vizinhos: (10+20+30+40)/4)",
             "",
             "b) Downsampling r=2, filtro linear 1/4[1 2 1] (separável), com zero-padding",
             "   Imagem filtrada:\n" + M(fil),
             f"   Amostrando 1 a cada {r}: g = {M(gd)}  ->  g(0,0) = {gd[0,0]:g}")


# ======================= Slide 29 =======================
def _bilinear(f, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    x1, y1 = min(x0 + 1, f.shape[1] - 1), min(y0 + 1, f.shape[0] - 1)
    return ((1 - dx) * (1 - dy) * f[y0, x0] + dx * (1 - dy) * f[y0, x1]
            + (1 - dx) * dy * f[y1, x0] + dx * dy * f[y1, x1])


def s29_1():
    f = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90.0]])
    s = 2
    xl, yl = 2, 3
    x, y = xl / s, yl / s
    v = _bilinear(f, x, y)
    return L("Mapeamento inverso: x = A^-1 x'  ->  (x, y) = (x'/s, y'/s)",
             f"(x, y) = ({x:g}, {y:g})",
             "Convenção: x = coluna, y = linha, f(x,y) = f[linha][coluna]",
             f"Vizinhos: f(1,1)=50, f(1,2)=80 (dx = 0, dy = 0.5)",
             f"Interpolação bilinear: f(x,y) = {v:g}")


def s29_2():
    X = np.array([0, 1, 2.0])
    d = np.array([1, 3, 2.0])
    Phi = np.abs(X[:, None] - X[None, :])
    w = np.linalg.solve(Phi, d)
    xq = 1.8
    val = (w * np.abs(xq - X)).sum()
    return L("φ(r) = |r|;  Φ w = d,  Φij = |xi - xj|", f"Φ =\n{M(Phi)}", f"w = {M(w)}",
             f"f(x) = Σ wk |x - xk|", f"f(1.8) = {w[0]:g}*1.8 + ({w[1]:g})*0.8 + {w[2]:g}*0.2 = {val:.4f}")


# ======================= Slide 28 =======================
def s28_1():
    f = np.array([[10, 20], [30, 40.0]])
    d = np.array([[12, 18], [29, 43.0]])
    c = 1
    E = (c * (f - d) ** 2).sum()
    return L("E_D = Σ c(i,j) (f(i,j) - d(i,j))²", f"(f - d)² =\n{M((f-d)**2)}", f"E_D = {E:g}")


def s28_2():
    f = np.array([10, 12, 15.0])
    dif = np.diff(f)
    return L("E1 = Σ (f(i+1) - f(i))²", f"diferenças = {M(dif)}", f"E1 = {(dif**2).sum():g}")


def s28_3():
    x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1])
    px = {k: (x == k).mean() for k in (0, 1)}
    py_x = {k: ((y == 0) & (x == k)).sum() / (x == k).sum() for k in (0, 1)}
    post = {k: py_x[k] * px[k] for k in (0, 1)}
    best = max(post, key=post.get)
    return L("Pixel central: y = 0.  MAP: x* = argmax p(y|x) p(x)",
             f"p(x=0) = {px[0]:.4f}, p(x=1) = {px[1]:.4f}",
             f"p(y=0|x=0) = {py_x[0]:.4f}, p(y=0|x=1) = {py_x[1]:.4f}",
             f"p(y=0|x=0)p(x=0) = {post[0]:.4f}   |   p(y=0|x=1)p(x=1) = {post[1]:.4f}",
             f"Rótulo MAP: x = {best}")


def s28_4():
    w, s = 1, 0.20
    d = 0
    viz = [1, 1, 1, 1]
    E = {}
    for fc in (0, 1):
        dado = (fc - d) ** 2
        suav = s * sum(w * (fc != v) for v in viz)
        E[fc] = dado + suav
    best = min(E, key=E.get)
    return L("E(f) = (f - d)² + s Σ_{vizinhos N4} w [f ≠ f_viz]  (vizinhos = 1)",
             f"E(f=0) = 0 + 0.2 * 4 = {E[0]:.2f}", f"E(f=1) = 1 + 0 = {E[1]:.2f}",
             f"Rótulo que minimiza a energia: f = {best}")


# ======================= Slide 31 =======================
def s31_1():
    A = [(1, 1), (2, 1), (1, 2)]
    B = [(4, 4), (5, 5), (4, 5)]
    pts = [(p, "A") for p in A] + [(p, "B") for p in B]
    x = np.array([3, 3.0])
    dist = [(np.linalg.norm(np.array(p) - x), c, p) for p, c in pts]
    dist.sort(key=lambda t: t[0])
    out = ["Distâncias de x = (3,3) (ordenadas):"]
    out += [f"  {p} classe {c}: {d:.4f}" for d, c, p in dist]
    for k in (1, 3):
        lim = dist[k - 1][0]
        viz = [t for t in dist if t[0] <= lim + 1e-9]
        votos = {c: sum(1 for t in viz if t[1] == c) for c in "AB"}
        txt = f"k={k}: vizinhos {[t[2] for t in viz]}, votos {votos}"
        if votos["A"] == votos["B"]:
            pes = {c: sum(1 / t[0] for t in viz if t[1] == c) for c in "AB"}
            r = max(pes, key=pes.get)
            txt += f"\n     empate (3 pontos a distância {lim:.3f}); desempate por voto ponderado 1/d: {pes[ 'A']:.3f} (A) x {pes['B']:.3f} (B)"
        else:
            r = max(votos, key=votos.get)
        out.append(txt + f"\n     -> classe {r}")
    return "\n".join(out)


def s31_2():
    w, b, x = np.array([1, -1.0]), 0.0, np.array([2, 1.0])
    z = w @ x + b
    p = 1 / (1 + np.exp(-z))
    return L("p(y=1|x) = σ(w·x + b)", f"z = {z:g}", f"p = 1/(1 + e^-{z:g}) = {p:.4f}")


def s31_3():
    w, b = np.array([1, 1.0]), -3
    pos = [(1, 4), (2, 2), (3, 3)]
    neg = [(0, 0), (0, 1), (2, 0)]
    dados = [(np.array(p, float), 1) for p in pos] + [(np.array(p, float), -1) for p in neg]
    marg = [(y * (w @ p + b), tuple(int(v) for v in p), y) for p, y in dados]
    mn = min(m[0] for m in marg)
    sv = [m for m in marg if np.isclose(m[0], mn)]
    x = np.array([2, 1.1])
    z = w @ x + b
    out = ["f(x) = w·x + b ; margem funcional yi f(xi):"]
    out += [f"  {p} (classe {y:+d}): f = {m*y:+g}, y f = {m:g}" for m, p, y in marg]
    out += [f"Vetores de suporte (yi f(xi) = {mn:g}): {[(p, y) for _, p, y in sv]}",
            f"f(2, 1.1) = {z:.2f}  ->  classe {'+1' if z > 0 else '-1'}"]
    return "\n".join(out)


# ======================= Slide 24 =======================
def s24_1():
    P = np.array([(1, 1), (2, 1), (4, 4), (5, 5)], float)
    C = np.array([(4, 3), (6, 6)], float)
    out = []
    for it in (1, 2):
        D = np.linalg.norm(P[:, None] - C[None], axis=2)
        lab = D.argmin(1)
        out.append(f"Iteração {it}: distâncias =\n{M(D)}\n  rótulos (0 = C1, 1 = C2): {lab.tolist()}")
        C = np.array([P[lab == k].mean(0) for k in range(2)])
        out.append(f"  novos centróides: C1 = {M(C[0])}, C2 = {M(C[1])}")
    return "\n".join(out)


def _gauss(x, mu, S):
    d = x - mu
    return np.exp(-0.5 * d @ np.linalg.inv(S) @ d) / (2 * np.pi * np.sqrt(np.linalg.det(S)))


def s24_2():
    X = np.array([(1, 0), (2, 1), (3, 3), (-1, -1)], float)
    pi = [0.4, 0.6]
    mu = [np.zeros(2), np.array([2, 2.0])]
    S = [np.eye(2), np.array([[2, 0], [0, 1.0]])]
    out = ["γik = πk N(xi|μk,Σk) / Σj πj N(xi|μj,Σj)"]
    for i, x in enumerate(X, 1):
        n = [pi[k] * _gauss(x, mu[k], S[k]) for k in range(2)]
        g = np.array(n) / sum(n)
        out.append(f"x{i} = {tuple(int(v) for v in x)}: π1N1 = {n[0]:.5f}, π2N2 = {n[1]:.5f}  ->  "
                   f"γ = ({g[0]:.4f}, {g[1]:.4f})  -> cluster {g.argmax()+1}")
    return "\n".join(out)


def s24_3():
    X = np.array([(2, 0), (0, 2), (3, 1), (1, 3)], float)
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / len(X)
    vals, vecs = np.linalg.eigh(C)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    return L(f"Média = {M(mu)}", f"Dados centrados:\n{M(Xc)}", f"Covariância (1/N):\n{M(C)}",
             f"Autovalores = {M(vals)}", f"Autovetores (colunas):\n{M(vecs)}",
             f"Variância total = {vals.sum():g}",
             f"Variância mantida pela 1ª eigenface = {vals[0]:g}/{vals.sum():g} = {vals[0]/vals.sum():.1%}",
             "(usando 1/(N-1) os autovalores mudam de escala, mas a proporção é a mesma)")


# ======================= Registro =======================
EXERCICIOS = [
    ("Slide 32", "1. Interseção de retas", "Retas a~ = (2,-1,3) e b~ = (1,1,-5). Calcule o ponto de interseção p = (x, y).", s32_1),
    ("Slide 32", "2. Transformação afim", "Aplique A = [2 0 1; 0 3 -1] no ponto p = (2, 1).", s32_2),
    ("Slide 32", "3. Transformação projetiva de reta", "Reta l~ = (2,-3,5), H~ = [1 0,5 2; 0 2 -1; 0 0,2 1]. Calcule l~'.", s32_3),
    ("Slide 32", "4. Quatérnio e rotação", "Eixo n = (0,6; 0,8; 0), ângulo 60°. Escreva o quatérnio unitário q e a matriz de rotação.", s32_4),
    ("Slide 36", "1. Projeção ortográfica escalada", "Ponto p = (3,-2,5), fator de escala s = 2.", s36_1),
    ("Slide 36", "2. Projeção perspectiva", "Ponto 3D p = (4,6,2). Calcule a projeção perspectiva no plano da imagem.", s36_2),
    ("Slide 36", "3. Distância focal pelo FOV", "Sensor W = 36 mm, campo de visão horizontal θH = 60°. Calcule f.", s36_3),
    ("Slide 36", "4. Distorção radial e pixel", "(xc,yc) = (0,5; 0,3), k1 = 0,10, k2 = 0,05; f = 800, cx = 320, cy = 240.", s36_4),
    ("Slide 40", "1. Phong especular", "Li = 200, ks = 0,5, θs = 60°, ke = 10. Intensidade especular refletida.", s40_1),
    ("Slide 40", "2. Reflexão difusa", "n = (0,0,1), vi = (√2/2, 0, √2/2), Li = 200, kd = 0,8.", s40_2),
    ("Slide 40", "3. Lente fina", "Objeto a z0 = 50 cm, plano da imagem a zi = 25 cm. Distância focal f?", s40_3),
    ("Slide 40", "4. Número f", "Lente com f = 50 mm e diâmetro de abertura d = 25 mm.", s40_4),
    ("Slide 42", "1. Operador Over", "Fundo = 100, frente = 200, α = 0,3. Cor resultante.", s42_1),
    ("Slide 42", "2. Corner, matrizes e imagem integral", "f = [1 2 3; 4 5 6; 7 8 9]. a) corner separável no pixel central; b) forma matricial (f, Hh, Hv); c) imagem integral e soma de (0,0) a (1,1).", s42_2),
    ("Slide 33", "1. Mediana, média α-cortada, mediana ponderada", "f = [10 2 3; 4 50 6; 7 8 9], W = [1 2 1; 2 4 2; 1 2 1]. Saída do pixel central.", s33_1),
    ("Slide 33", "2. Upsampling e downsampling", "f = [10 20; 30 40]. a) 4x4 (r=2) com filtro linear, g(1,1); b) 1x1 (r=2) com zero-padding.", s33_2),
    ("Slide 29", "1. Escala + interpolação bilinear", "s = 2, R = I, t = 0, f = [10 20 30; 40 50 60; 70 80 90]. Mapeamento inverso de (x',y') = (2,3).", s29_1),
    ("Slide 29", "2. Interpolação RBF", "x = {0,1,2}, d = {1,3,2}, φ = distância euclidiana. Pesos w e f(1.8).", s29_2),
    ("Slide 28", "1. Energia de dados", "f = [10 20; 30 40], d = [12 18; 29 43], c = 1.", s28_1),
    ("Slide 28", "2. Energia de suavidade", "f = [10, 12, 15]. E1 = Σ (f(i+1) - f(i))².", s28_2),
    ("Slide 28", "3. MAP em imagem binária 1D", "x = (1,1,1,0,0,0,0,0,0), y = (1,1,0,1,0,0,0,0,1). Rótulo do pixel central (y = 0).", s28_3),
    ("Slide 28", "4. Energia MRF 3x3", "Pixel central observado 0, vizinhos N4 observados 1, w = 1, s = 0,20.", s28_4),
    ("Slide 31", "1. k-NN", "Classe A: (1,1),(2,1),(1,2). Classe B: (4,4),(5,5),(4,5). Classifique x = (3,3) com k = 1 e k = 3.", s31_1),
    ("Slide 31", "2. Regressão logística", "w = (1,-1), b = 0, x = (2,1). Probabilidade da classe positiva.", s31_2),
    ("Slide 31", "3. SVM", "w = (1,1), b = -3. +1: (1,4),(2,2),(3,3); -1: (0,0),(0,1),(2,0). Vetores de suporte e classe de (2, 1.1).", s31_3),
    ("Slide 24", "1. k-means", "Pontos (1,1),(2,1),(4,4),(5,5); C1 = (4,3), C2 = (6,6). Centróides após 2 iterações.", s24_1),
    ("Slide 24", "2. GMM (E-step)", "x1..x4 = (1,0),(2,1),(3,3),(-1,-1); π = (0,4; 0,6), μ1 = (0,0), μ2 = (2,2), Σ1 = I, Σ2 = [2 0; 0 1].", s24_2),
    ("Slide 24", "3. PCA / eigenfaces", "Pontos (2,0),(0,2),(3,1),(1,3). Variância mantida pela 1ª eigenface.", s24_3),
]


def resolver(i):
    slide, titulo, enunciado, fn = EXERCICIOS[i]
    return f"{slide} - {titulo}\n\nEnunciado:\n{enunciado}\n\nResolução:\n{fn()}\n"
