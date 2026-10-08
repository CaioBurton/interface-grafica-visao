"""Aba "Blending (Pirâmides)": mistura de duas imagens com pirâmides Gaussiana/Laplaciana (Atividade 02)."""
import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw

from vc import piramides

PASTA_IMAGENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imagens")
PADROES = {"a": "512_burt_apple.png", "b": "512_burt_orange.png", "mascara": "512_mask.jpg"}
ROTULOS = {"a": "Imagem A (onde máscara = 1)", "b": "Imagem B (onde máscara = 0)", "mascara": "Máscara"}
TIPOS = [("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
VISOES = ["Comparação", "Blending", "Justaposição", "Zoom da emenda", "Pirâmides"]
MASCARA_ARQUIVO = "Arquivo"
MASCARA_ESQ = "Metade esquerda = A"
MASCARA_DIR = "Metade direita = A"


def _pil(arr):
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _rotular(img, texto):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 8 + 7 * len(texto), 18), fill=(0, 0, 0))
    d.text((4, 3), texto, fill=(255, 255, 255))
    return img


def _grade(imgs, colunas):
    w, h = imgs[0].size
    linhas = -(-len(imgs) // colunas)
    tela = Image.new("RGB", (w * colunas, h * linhas), "white")
    for i, im in enumerate(imgs):
        tela.paste(im, ((i % colunas) * w, (i // colunas) * h))
    return tela


class BlendingTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.caminhos = {k: None for k in PADROES}
        self.rotulos_arquivo = {}
        self.resultados = {}   # visão -> imagem PIL
        self._ctk_image = None
        self._atual = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_painel()
        self.image_label = ctk.CTkLabel(self, text="Escolha as imagens e clique em Aplicar blending")
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_label.bind("<Configure>", lambda _e: self._exibir())
        self.status = ctk.CTkLabel(self, text="", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))
        self.usar_sugeridas()

    # ---------- interface ----------
    def _build_painel(self):
        p = ctk.CTkScrollableFrame(self, width=270, corner_radius=0)
        p.grid(row=0, column=0, sticky="ns")

        def secao(t):
            ctk.CTkLabel(p, text=t, anchor="w", text_color="gray").pack(fill="x", padx=10, pady=(14, 2))

        secao("IMAGENS")
        for chave in PADROES:
            ctk.CTkButton(p, text=f"Escolher {ROTULOS[chave]}",
                          command=lambda c=chave: self.escolher(c)).pack(fill="x", padx=10, pady=(3, 0))
            lbl = ctk.CTkLabel(p, text="-", anchor="w", text_color="gray", font=ctk.CTkFont(size=11))
            lbl.pack(fill="x", padx=12)
            self.rotulos_arquivo[chave] = lbl
        ctk.CTkButton(p, text="Usar imagens sugeridas (maçã + laranja)", fg_color="gray30", hover_color="gray40",
                      command=self.usar_sugeridas).pack(fill="x", padx=10, pady=(8, 3))

        secao("MÁSCARA")
        self.modo_mascara = ctk.CTkOptionMenu(p, values=[MASCARA_ARQUIVO, MASCARA_ESQ, MASCARA_DIR])
        self.modo_mascara.pack(fill="x", padx=10, pady=3)

        secao("PIRÂMIDE")
        linha = ctk.CTkFrame(p, fg_color="transparent")
        linha.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(linha, text="Níveis", width=90, anchor="w").pack(side="left")
        self.niveis = ctk.CTkEntry(linha)
        self.niveis.insert(0, "5")
        self.niveis.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(p, text="Aplicar blending", command=self.aplicar).pack(fill="x", padx=10, pady=(10, 3))

        secao("VISUALIZAÇÃO")
        self.visao = ctk.CTkOptionMenu(p, values=VISOES, command=lambda _v: self._exibir())
        self.visao.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(p, text="Salvar visualização", command=self.salvar).pack(fill="x", padx=10, pady=(10, 3))

    # ---------- entrada ----------
    def _definir(self, chave, caminho):
        self.caminhos[chave] = caminho
        self.rotulos_arquivo[chave].configure(text=os.path.basename(caminho) if caminho else "-")

    def escolher(self, chave):
        caminho = filedialog.askopenfilename(title=ROTULOS[chave], filetypes=TIPOS)
        if caminho:
            self._definir(chave, caminho)
            if chave == "mascara":
                self.modo_mascara.set(MASCARA_ARQUIVO)

    def usar_sugeridas(self):
        for chave, nome in PADROES.items():
            c = os.path.normpath(os.path.join(PASTA_IMAGENS, nome))
            self._definir(chave, c if os.path.exists(c) else None)
        self.modo_mascara.set(MASCARA_ARQUIVO)
        if all(self.caminhos.values()):
            self.aplicar()

    # ---------- processamento ----------
    def _mascara(self, shape):
        h, w = shape
        modo = self.modo_mascara.get()
        if modo == MASCARA_ARQUIVO:
            if not self.caminhos["mascara"]:
                raise ValueError("Escolha a máscara ou use uma das máscaras automáticas.")
            m = np.asarray(Image.open(self.caminhos["mascara"]).convert("L").resize((w, h), Image.NEAREST), float)
            return (m / 255 > 0.5).astype(float)
        m = np.zeros((h, w))
        m[:, :w // 2] = 1
        return m if modo == MASCARA_ESQ else 1 - m

    def aplicar(self):
        try:
            if not (self.caminhos["a"] and self.caminhos["b"]):
                raise ValueError("Escolha as imagens A e B.")
            niveis = int(self.niveis.get())
            A = Image.open(self.caminhos["a"]).convert("RGB")
            a = np.asarray(A, float)
            b = np.asarray(Image.open(self.caminhos["b"]).convert("RGB").resize(A.size, Image.LANCZOS), float)
            m = self._mascara(a.shape[:2])
            maximo = int(np.log2(min(a.shape[:2]) / 4)) + 1
            if not 1 <= niveis <= maximo:
                raise ValueError(f"Níveis deve estar entre 1 e {maximo} para esta imagem.")
            mix = np.clip(piramides.blending(a, b, m, niveis), 0, 255)
            cola = piramides.justaposicao(a, b, m)
        except Exception as exc:
            return messagebox.showerror("Erro", str(exc))

        pa, pb, pm, pc, px = _pil(a), _pil(b), _pil(m * 255).convert("RGB"), _pil(cola), _pil(mix)
        diff = np.abs(mix - cola).mean(axis=2)
        pd = _pil(255 * (diff / diff.max() if diff.max() > 0 else diff)).convert("RGB")
        h, w = m.shape
        x = int(np.argmax(np.abs(np.diff(m.mean(axis=0))))) + 1 if m.min() != m.max() else w // 2
        box = (max(0, x - w // 4), h // 4, min(w, x + w // 4), 3 * h // 4)

        def zoom(im, t):
            return _rotular(im.crop(box).resize((w, h), Image.NEAREST), t)

        self.resultados = {
            "Comparação": _grade([_rotular(pa, "A"), _rotular(pb, "B"), _rotular(pm, "Máscara"),
                                  _rotular(pc, "Justaposição direta"),
                                  _rotular(px, f"Blending ({niveis} níveis)"),
                                  _rotular(pd, "|blending - justaposição|")], 3),
            "Blending": px, "Justaposição": pc,
            "Zoom da emenda": _grade([zoom(pc, "Justaposição"), zoom(px, "Blending")], 2),
            "Pirâmides": self._mosaico(a, m, niveis),
        }
        self.status.configure(text=f"Blending com {niveis} níveis  |  {w}x{h}")
        self._exibir()

    @staticmethod
    def _mosaico(a, m, niveis):
        """Pirâmide Gaussiana da máscara (cima) e Laplaciana de A (baixo), níveis lado a lado."""
        gm, la = piramides.piramide_gaussiana(m, niveis), piramides.piramide_laplaciana(a, niveis)
        linhas = [[_pil(g * 255).convert("RGB") for g in gm],
                  [_pil(l if i == niveis - 1 else l + 128) for i, l in enumerate(la)]]
        h = linhas[0][0].height
        tela = Image.new("RGB", (sum(i.width for i in linhas[0]), 2 * h), (60, 60, 60))
        for fila, linha in enumerate(linhas):
            x = 0
            for i, im in enumerate(linha):
                tela.paste(im, (x, fila * h))
                x += im.width
        return tela

    # ---------- exibição ----------
    def _exibir(self):
        img = self.resultados.get(self.visao.get())
        if img is None:
            return
        cw, ch = max(self.image_label.winfo_width(), 1), max(self.image_label.winfo_height(), 1)
        esc = min(cw / img.width, ch / img.height)
        tam = (max(1, int(img.width * esc)), max(1, int(img.height * esc)))
        self._atual = img
        self._ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=tam)
        self.image_label.configure(image=self._ctk_image, text="")

    def salvar(self):
        if self._atual is None:
            return messagebox.showinfo("Aviso", "Aplique o blending primeiro.")
        caminho = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if caminho:
            self._atual.save(caminho)
