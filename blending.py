"""Atividade 02 - blending de imagens com pirâmides (maçã + laranja).

Uso:  python blending.py [pasta_imagens] [pasta_saida] [niveis]
Gera resultado_blending.png, resultado_justaposicao.png, comparacao.png e piramides.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vc import piramides


def carregar(pasta, nome, modo):
    return np.asarray(Image.open(os.path.join(pasta, nome)).convert(modo), float)


def salvar(arr, caminho):
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(caminho)


def main(pasta, saida, niveis):
    os.makedirs(saida, exist_ok=True)
    a, b = carregar(pasta, "512_burt_apple.png", "RGB"), carregar(pasta, "512_burt_orange.png", "RGB")
    m = carregar(pasta, "512_mask.jpg", "L") / 255.0
    m = (m > 0.5).astype(float)   # máscara binária (o JPEG introduz ruído de compressão)

    mix = np.clip(piramides.blending(a, b, m, niveis), 0, 255)
    cola = piramides.justaposicao(a, b, m)
    salvar(mix, os.path.join(saida, "resultado_blending.png"))
    salvar(cola, os.path.join(saida, "resultado_justaposicao.png"))

    # reconstrução sem mistura: confere a pirâmide Laplaciana
    erro = np.abs(piramides.colapsar(piramides.piramide_laplaciana(a, niveis)) - a).max()
    print(f"erro máximo de reconstrução da pirâmide Laplaciana: {erro:.2e}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for x, im, t in zip(ax.ravel(), (a, b, m, cola, mix), ("Maçã", "Laranja", "Máscara",
                                                            "Justaposição direta", f"Blending ({niveis} níveis)")):
        x.imshow(im.astype(np.uint8) if im.ndim == 3 else im, cmap="gray"); x.set_title(t); x.axis("off")
    ax[1, 2].imshow(np.abs(mix - cola).mean(axis=2), cmap="magma"); ax[1, 2].set_title("|blending − justaposição|")
    ax[1, 2].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(saida, "comparacao.png"), dpi=100); plt.close(fig)

    # zoom na região da emenda
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for x, im, t in zip(ax, (cola, mix), ("Justaposição (emenda visível)", "Blending (emenda suave)")):
        x.imshow(im[150:360, 156:356].astype(np.uint8)); x.set_title(t); x.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(saida, "zoom_emenda.png"), dpi=100); plt.close(fig)

    # pirâmides
    gm, la = piramides.piramide_gaussiana(m, niveis), piramides.piramide_laplaciana(a, niveis)
    fig, ax = plt.subplots(2, niveis, figsize=(3 * niveis, 6))
    for i in range(niveis):
        ax[0, i].imshow(gm[i], cmap="gray", vmin=0, vmax=1); ax[0, i].set_title(f"Gaussiana máscara, nível {i}")
        l = la[i] if i == niveis - 1 else la[i] + 128
        ax[1, i].imshow(np.clip(l, 0, 255).astype(np.uint8)); ax[1, i].set_title(f"Laplaciana maçã, nível {i}")
        ax[0, i].axis("off"); ax[1, i].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(saida, "piramides.png"), dpi=80); plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "..", "imagens"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "resultados_blending"),
         int(sys.argv[3]) if len(sys.argv) > 3 else 5)
