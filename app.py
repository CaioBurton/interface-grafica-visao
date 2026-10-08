"""Interface gráfica de visão computacional (CustomTkinter) sobre a biblioteca `vc`."""
import ast
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

from gui_blending import BlendingTab
from gui_formacao import FormacaoImagemTab
from registro import FUNCOES
from vc import aprendizado, filtros, otimizacao, transformacoes

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FILTROS = ["Média (box)", "Média (imagem integral)", "Gaussiano", "Corner (separável)",
           "Mediana", "Média α-cortada", "Mediana ponderada"]


def formatar(res):
    if isinstance(res, tuple):
        return "\n\n".join(formatar(r) for r in res)
    if isinstance(res, np.ndarray) or isinstance(res, (list, float, np.floating)):
        return np.array2string(np.asarray(res), precision=4, suppress_small=True)
    return str(res)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Visão Computacional")
        self.geometry("1200x780")
        self.minsize(900, 600)

        self.original = None   # imagem original (PIL), nunca modificada
        self.current = None    # imagem atual (PIL)
        self._ctk_image = None
        self.entries = {}

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        tab_img = self.tabs.add("Imagem")
        tab_fun = self.tabs.add("Funções")

        tab_img.grid_columnconfigure(1, weight=1)
        tab_img.grid_rowconfigure(0, weight=1)
        self._build_sidebar(tab_img)

        self.image_label = ctk.CTkLabel(tab_img, text="Abra uma imagem para começar", font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_label.bind("<Configure>", lambda _e: self.show())

        self.status = ctk.CTkLabel(tab_img, text="Nenhuma imagem carregada.", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

        self._build_funcoes(tab_fun)

        FormacaoImagemTab(self.tabs.add("Formação de Imagem")).pack(fill="both", expand=True)
        BlendingTab(self.tabs.add("Blending (Pirâmides)")).pack(fill="both", expand=True)

        self.bind("<Control-o>", lambda _e: self.load_image())
        self.bind("<Control-s>", lambda _e: self.save_image())
        self.bind("<Control-r>", lambda _e: self.restore_image())

    # ---------- construção da interface ----------
    def _secao(self, parent, titulo):
        ctk.CTkLabel(parent, text=titulo, anchor="w", text_color="gray").pack(fill="x", padx=10, pady=(14, 2))

    def _botao(self, parent, texto, cmd, **kw):
        ctk.CTkButton(parent, text=texto, command=cmd, **kw).pack(fill="x", padx=10, pady=3)

    def _campo(self, parent, chave, rotulo, padrao):
        linha = ctk.CTkFrame(parent, fg_color="transparent")
        linha.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(linha, text=rotulo, width=90, anchor="w").pack(side="left")
        e = ctk.CTkEntry(linha)
        e.insert(0, padrao)
        e.pack(side="left", fill="x", expand=True)
        self.entries[chave] = e

    def _build_sidebar(self, parent):
        side = ctk.CTkScrollableFrame(parent, width=250, corner_radius=0)
        side.grid(row=0, column=0, sticky="ns")

        self._secao(side, "ARQUIVO")
        self._botao(side, "Abrir imagem", self.load_image)
        self._botao(side, "Salvar imagem", self.save_image)

        self._secao(side, "BÁSICO")
        self._botao(side, "Zerar intensidade", self.zero_image)
        self._botao(side, "Restaurar original", self.restore_image, fg_color="gray30", hover_color="gray40")
        self._botao(side, "Escala de cinza", self.op_cinza)

        self._secao(side, "FILTROS")
        self.filtro_menu = ctk.CTkOptionMenu(side, values=FILTROS)
        self.filtro_menu.pack(fill="x", padx=10, pady=3)
        self._campo(side, "k", "Janela k", "3")
        self._campo(side, "alpha", "α (cortada)", "2")
        self._botao(side, "Aplicar filtro", self.op_filtro)

        self._secao(side, "TRANSFORMAÇÃO GEOMÉTRICA")
        self._campo(side, "s", "Escala", "1.0")
        self._campo(side, "theta", "Ângulo (°)", "0")
        self._campo(side, "tx", "Transl. x", "0")
        self._campo(side, "ty", "Transl. y", "0")
        self.interp_menu = ctk.CTkOptionMenu(side, values=["bilinear", "vizinho"])
        self.interp_menu.pack(fill="x", padx=10, pady=3)
        self._botao(side, "Aplicar transformação", self.op_transformar)

        self._secao(side, "REAMOSTRAGEM")
        self._campo(side, "r", "Fator r", "2")
        self._botao(side, "Aumentar resolução", lambda: self.op_reamostrar(True))
        self._botao(side, "Diminuir resolução", lambda: self.op_reamostrar(False))

        self._secao(side, "DISTORÇÃO RADIAL")
        self._campo(side, "k1", "k1", "0.1")
        self._campo(side, "k2", "k2", "0.0")
        self._botao(side, "Corrigir distorção", self.op_distorcao)

        self._secao(side, "COMPOSIÇÃO (OVER)")
        self._campo(side, "over_a", "α", "0.5")
        self._botao(side, "Compor com imagem de frente...", self.op_over)

        self._secao(side, "IMAGEM INTEGRAL")
        self._campo(side, "ret", "i0,j0,i1,j1", "0,0,10,10")
        self._botao(side, "Soma no retângulo", self.op_soma_retangulo)

        self._secao(side, "SEGMENTAÇÃO / MRF")
        self._campo(side, "kc", "Clusters k", "3")
        self._botao(side, "Segmentar (k-means)", self.op_kmeans)
        self._campo(side, "icm_s", "Suavidade s", "0.3")
        self._botao(side, "Binarizar + limpar (ICM)", self.op_icm)

        self._secao(side, "TEMA")
        ctk.CTkOptionMenu(side, values=["Dark", "Light", "System"],
                          command=lambda m: ctk.set_appearance_mode(m)).pack(fill="x", padx=10, pady=(3, 14))

    def _build_funcoes(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        lista = ctk.CTkScrollableFrame(parent, width=260)
        lista.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=4)
        cat = None
        for i, (categoria, nome, _fn, _p) in enumerate(FUNCOES):
            if categoria != cat:
                cat = categoria
                ctk.CTkLabel(lista, text=categoria.upper(), anchor="w", text_color="gray").pack(fill="x", pady=(10, 2))
            ctk.CTkButton(lista, text=nome, anchor="w", height=26, command=lambda i=i: self.selecionar_funcao(i)
                          ).pack(fill="x", pady=1)

        direita = ctk.CTkFrame(parent, fg_color="transparent")
        direita.grid(row=0, column=1, sticky="nsew")
        direita.grid_columnconfigure(0, weight=1)
        direita.grid_rowconfigure(3, weight=1)
        self.fun_titulo = ctk.CTkLabel(direita, text="Selecione uma função", anchor="w",
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.fun_titulo.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        self.fun_doc = ctk.CTkLabel(direita, text="", anchor="w", justify="left", wraplength=700, text_color="gray")
        self.fun_doc.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.fun_params = ctk.CTkFrame(direita)
        self.fun_params.grid(row=2, column=0, sticky="ew")
        self.fun_params.grid_columnconfigure(1, weight=1)
        self.fun_saida = ctk.CTkTextbox(direita, font=ctk.CTkFont(family="Consolas", size=13))
        self.fun_saida.grid(row=3, column=0, sticky="nsew", pady=8)
        self.fun_atual = None
        self.fun_campos = []

    # ---------- aba Funções ----------
    def selecionar_funcao(self, i):
        _cat, nome, fn, params = FUNCOES[i]
        self.fun_atual = i
        self.fun_titulo.configure(text=nome)
        doc = (fn.__doc__ or "").strip()
        self.fun_doc.configure(text=doc)
        for w in self.fun_params.winfo_children():
            w.destroy()
        self.fun_campos = []
        for r, (pnome, padrao) in enumerate(params):
            ctk.CTkLabel(self.fun_params, text=pnome, anchor="w", width=110).grid(row=r, column=0, padx=8, pady=3)
            e = ctk.CTkEntry(self.fun_params)
            e.insert(0, padrao)
            e.grid(row=r, column=1, sticky="ew", padx=8, pady=3)
            self.fun_campos.append(e)
        ctk.CTkButton(self.fun_params, text="Calcular", command=self.calcular_funcao).grid(
            row=len(params), column=0, columnspan=2, pady=8)
        self.fun_saida.delete("1.0", "end")

    def calcular_funcao(self):
        if self.fun_atual is None:
            return
        _cat, _nome, fn, params = FUNCOES[self.fun_atual]
        try:
            args = [ast.literal_eval(e.get().strip()) for e in self.fun_campos]
            res = fn(*args)
            texto = formatar(res)
        except Exception as exc:
            texto = f"Erro: {exc}"
        self.fun_saida.delete("1.0", "end")
        self.fun_saida.insert("end", texto)

    # ---------- arquivo e operações básicas ----------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
        except Exception as exc:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{exc}")
            return
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        self.original = img
        self.current = img.copy()
        self.status.configure(text=f"{path}  |  {img.width}x{img.height}  |  modo {img.mode}")
        self.show()

    def save_image(self):
        if not self._has_image():
            return
        path = filedialog.asksaveasfilename(
            title="Salvar imagem",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif *.tiff")],
        )
        if not path:
            return
        try:
            self.current.save(path)
        except Exception as exc:
            messagebox.showerror("Erro", f"Não foi possível salvar:\n{exc}")
            return
        self.status.configure(text=f"Imagem salva em {path}")

    def zero_image(self):
        """Atribui intensidade zero a todos os pixels."""
        if not self._has_image():
            return
        self.current = Image.new(self.current.mode, self.current.size, 0)
        self.status.configure(text="Todos os pixels com intensidade 0.")
        self.show()

    def restore_image(self):
        """Retorna à imagem original."""
        if not self._has_image():
            return
        self.current = self.original.copy()
        self.status.configure(text="Imagem original restaurada.")
        self.show()

    # ---------- operações com a biblioteca ----------
    def _num(self, chave, tipo=float):
        return tipo(self.entries[chave].get().replace(",", "."))

    def _array(self):
        return np.asarray(self.current, dtype=float)

    def _aplicar(self, fn, msg, normalizar=False):
        """Executa fn(array) -> array e exibe o resultado como imagem."""
        if not self._has_image():
            return
        try:
            res = np.asarray(fn(self._array()), float)
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            return
        if normalizar:
            res = np.abs(res)
            res = res * (255 / res.max()) if res.max() > 0 else res
        self.current = Image.fromarray(np.clip(np.rint(res), 0, 255).astype(np.uint8))
        self.status.configure(text=f"{msg}  |  {self.current.width}x{self.current.height}")
        self.show()

    def op_cinza(self):
        if not self._has_image():
            return
        self.current = self.current.convert("L")
        self.status.configure(text="Convertida para escala de cinza.")
        self.show()

    def op_filtro(self):
        nome = self.filtro_menu.get()
        try:
            k, alpha = self._num("k", int), self._num("alpha", int)
        except ValueError:
            return messagebox.showerror("Erro", "Parâmetros inválidos.")
        if k % 2 == 0 and nome != "Corner (separável)":
            return messagebox.showerror("Erro", "A janela k deve ser ímpar.")
        # modo "reflect" evita escurecimento nas bordas
        def f(a):
            if nome == "Média (box)":
                return filtros.convolucao(a, filtros.nucleo_media(k), "reflect")
            if nome == "Média (imagem integral)":
                return filtros.filtro_media_integral(a, k)
            if nome == "Gaussiano":
                return filtros.convolucao(a, filtros.nucleo_gaussiano(k), "reflect")
            if nome == "Corner (separável)":
                return filtros.filtro_separavel(a, filtros.H_CORNER, filtros.H_CORNER, "reflect")
            if nome == "Mediana":
                return filtros.mediana(a, k, "reflect")
            if nome == "Média α-cortada":
                return filtros.media_alfa_cortada(a, k, alpha, "reflect")
            return filtros.mediana_ponderada(a, filtros.nucleo_binomial(k) * (2 ** (2 * (k - 1))), "reflect")
        self._aplicar(f, f"Filtro: {nome}", normalizar=(nome == "Corner (separável)"))

    def op_transformar(self):
        try:
            s, th, tx, ty = (self._num(c) for c in ("s", "theta", "tx", "ty"))
        except ValueError:
            return messagebox.showerror("Erro", "Parâmetros inválidos.")
        interp = self.interp_menu.get()
        if not self._has_image():
            return
        M = transformacoes.similaridade_centrada(self._array().shape[:2], s, th, (tx, ty))
        self._aplicar(lambda a: transformacoes.transformar_imagem(a, M, interpolacao=interp),
                      f"Similaridade s={s:g}, θ={th:g}°, t=({tx:g},{ty:g})")

    def op_reamostrar(self, aumentar):
        try:
            r = self._num("r", int)
            if r < 2:
                raise ValueError
        except ValueError:
            return messagebox.showerror("Erro", "O fator r deve ser um inteiro >= 2.")
        fn = transformacoes.aumentar_resolucao if aumentar else transformacoes.diminuir_resolucao
        self._aplicar(lambda a: fn(a, r), f"{'Aumento' if aumentar else 'Redução'} de resolução (r={r})")

    def op_distorcao(self):
        try:
            k1, k2 = self._num("k1"), self._num("k2")
        except ValueError:
            return messagebox.showerror("Erro", "Parâmetros inválidos.")
        self._aplicar(lambda a: transformacoes.corrigir_distorcao_radial(a, k1, k2),
                      f"Distorção radial k1={k1:g}, k2={k2:g}")

    def op_over(self):
        if not self._has_image():
            return
        try:
            alpha = self._num("over_a")
        except ValueError:
            return messagebox.showerror("Erro", "α inválido.")
        path = filedialog.askopenfilename(title="Imagem de frente",
                                          filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif")])
        if not path:
            return
        try:
            frente = Image.open(path).convert(self.current.mode).resize(self.current.size)
        except Exception as exc:
            return messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{exc}")
        F = np.asarray(frente, float)
        self._aplicar(lambda a: transformacoes.composicao_over(F, a, alpha), f"Over (α={alpha:g})")

    def op_soma_retangulo(self):
        if not self._has_image():
            return
        try:
            i0, j0, i1, j1 = (int(v) for v in self.entries["ret"].get().split(","))
            g = np.asarray(self.current.convert("L"), float)
            if not (0 <= i0 <= i1 < g.shape[0] and 0 <= j0 <= j1 < g.shape[1]):
                raise ValueError("Retângulo fora da imagem.")
            soma = filtros.soma_retangulo(filtros.imagem_integral(g), i0, j0, i1, j1)
        except Exception as exc:
            return messagebox.showerror("Erro", f"Use 'i0,j0,i1,j1' (linha, coluna) dentro da imagem.\n{exc}")
        self.status.configure(text=f"Soma dos pixels (cinza) em [{i0},{j0}]-[{i1},{j1}] = {soma:g}")

    def op_kmeans(self):
        try:
            k = self._num("kc", int)
        except ValueError:
            return messagebox.showerror("Erro", "k inválido.")
        self._aplicar(lambda a: aprendizado.segmentar_imagem_kmeans(a, k), f"K-means (k={k})")

    def op_icm(self):
        try:
            s = self._num("icm_s")
        except ValueError:
            return messagebox.showerror("Erro", "s inválido.")

        def f(a):
            g = a if a.ndim == 2 else a.mean(axis=2)
            return otimizacao.restaurar_icm(g > g.mean(), s=s) * 255
        self._aplicar(f, f"Binarização + ICM (s={s:g})")

    # ---------- utilitários ----------
    def _has_image(self):
        if self.current is None:
            messagebox.showinfo("Aviso", "Carregue uma imagem primeiro.")
            return False
        return True

    def show(self):
        """Exibe a imagem atual ajustada ao tamanho da área central."""
        if self.current is None:
            return
        cw, ch = max(self.image_label.winfo_width(), 1), max(self.image_label.winfo_height(), 1)
        scale = min(cw / self.current.width, ch / self.current.height, 1.0)
        size = (max(int(self.current.width * scale), 1), max(int(self.current.height * scale), 1))
        self._ctk_image = ctk.CTkImage(light_image=self.current, dark_image=self.current, size=size)
        self.image_label.configure(image=self._ctk_image, text="")


if __name__ == "__main__":
    App().mainloop()
