"""Aba "Formação de Imagem": projeção perspectiva x ortográfica escalada de um cubo."""
import math

import customtkinter as ctk
import numpy as np

from vc import formacao_imagem as fi

PERSPECTIVA, ORTOGRAFICA = "Perspectiva", "Ortográfica escalada"

# Faixa do slider de cada parâmetro (o campo de texto aceita qualquer valor).
LIMITES = {"fx": (100, 3000), "fy": (100, 3000), "cx": (0, 1280), "cy": (0, 720), "skew": (-500, 500),
           "escala": (10, 500), "tx": (-10, 10), "ty": (-10, 10), "tz": (0, 20),
           "rx": (-180, 180), "ry": (-180, 180), "rz": (-180, 180)}


class FormacaoImagemTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.vars = {}
        self._atualizando = False   # evita recursão ao sincronizar fx=fy
        self._ctk_image = None
        self._ultima = None         # última imagem PIL gerada

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_painel()

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_label.bind("<Configure>", lambda _e: self._exibir())
        self._arrasto = None   # (x, y) do último ponto do arrasto com o mouse
        self._esc = 1.0        # fator de exibição (pixels da tela / pixels do sensor)
        for ev, fn in (("<ButtonPress-1>", self._mouse_inicio), ("<ButtonPress-3>", self._mouse_inicio),
                       ("<B1-Motion>", self._mouse_gira), ("<B3-Motion>", self._mouse_translada),
                       ("<ButtonRelease-1>", self._mouse_fim), ("<ButtonRelease-3>", self._mouse_fim),
                       ("<MouseWheel>", self._mouse_roda)):
            self.image_label.bind(ev, fn)
        self.status = ctk.CTkLabel(self, text="", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

        self._trocar_modelo()

    # ---------- interface ----------
    def _secao(self, titulo, parent=None):
        ctk.CTkLabel(parent or self.painel, text=titulo, anchor="w", text_color="gray").pack(
            fill="x", padx=10, pady=(14, 2))

    def _campo(self, parent, chave, rotulo, valor):
        minimo, maximo = LIMITES[chave]
        bloco = ctk.CTkFrame(parent, fg_color="transparent")
        bloco.pack(fill="x", padx=10, pady=2)
        linha = ctk.CTkFrame(bloco, fg_color="transparent")
        linha.pack(fill="x")
        ctk.CTkLabel(linha, text=rotulo, width=90, anchor="w").pack(side="left")
        var = ctk.StringVar(value=f"{valor:g}")
        ctk.CTkEntry(linha, textvariable=var).pack(side="left", fill="x", expand=True)
        slider = ctk.CTkSlider(bloco, from_=minimo, to=maximo, height=14,
                               command=lambda v: var.set(f"{round(v, 2):g}"))
        slider.set(valor)
        slider.pack(fill="x", pady=(2, 0))

        def ao_digitar(*_):
            try:
                slider.set(min(max(self._numero(var.get()), minimo), maximo))
            except ValueError:
                pass
            self._ao_alterar(chave)
        var.trace_add("write", ao_digitar)
        self.vars[chave] = var

    def _build_painel(self):
        self.painel = ctk.CTkScrollableFrame(self, width=270, corner_radius=0)
        self.painel.grid(row=0, column=0, sticky="ns")

        self._secao("MODELO DE PROJEÇÃO")
        self.modelo = ctk.CTkOptionMenu(self.painel, values=[PERSPECTIVA, ORTOGRAFICA],
                                        command=lambda _m: self._trocar_modelo())
        self.modelo.pack(fill="x", padx=10, pady=3)

        self._secao("RESOLUÇÃO DO SENSOR")
        self.resolucao = ctk.CTkOptionMenu(self.painel, values=list(fi.RESOLUCOES),
                                           command=lambda _m: self._atualizar())
        self.resolucao.pack(fill="x", padx=10, pady=3)

        # Parâmetros específicos de cada modelo (apenas um fica visível por vez)
        self.area_modelo = ctk.CTkFrame(self.painel, fg_color="transparent")
        self.area_modelo.pack(fill="x")

        self.frame_persp = ctk.CTkFrame(self.area_modelo, fg_color="transparent")
        self._secao("PARÂMETROS INTRÍNSECOS", self.frame_persp)
        p = fi.PADRAO_PERSPECTIVA
        for chave, rot in (("fx", "fx"), ("fy", "fy"), ("cx", "cx"), ("cy", "cy"), ("skew", "skew")):
            self._campo(self.frame_persp, chave, rot, p[chave])
        self.igualar = ctk.CTkCheckBox(self.frame_persp, text="Manter fx = fy", command=self._ao_igualar)
        self.igualar.select()
        self.igualar.pack(anchor="w", padx=10, pady=6)
        ctk.CTkLabel(self.frame_persp, text="Matriz K", anchor="w", text_color="gray").pack(
            fill="x", padx=10, pady=(6, 0))
        self.k_label = ctk.CTkLabel(self.frame_persp, text="", justify="left", anchor="w",
                                    font=ctk.CTkFont(family="Consolas", size=12))
        self.k_label.pack(fill="x", padx=10)

        self.frame_orto = ctk.CTkFrame(self.area_modelo, fg_color="transparent")
        self._secao("PARÂMETROS DA ORTOGRÁFICA", self.frame_orto)
        self._campo(self.frame_orto, "escala", "Escala", fi.PADRAO_ESCALA)

        self._secao("TRANSLAÇÃO (extrínsecos)")
        e = fi.PADRAO_EXTRINSECOS
        for chave, rot in (("tx", "tx"), ("ty", "ty"), ("tz", "tz")):
            self._campo(self.painel, chave, rot, e[chave])
        self._secao("ROTAÇÃO em graus (extrínsecos)")
        for chave, rot in (("rx", "Rot. X (°)"), ("ry", "Rot. Y (°)"), ("rz", "Rot. Z (°)")):
            self._campo(self.painel, chave, rot, e[chave])

        ctk.CTkButton(self.painel, text="Restaurar valores iniciais", fg_color="gray30",
                      hover_color="gray40", command=self.restaurar).pack(fill="x", padx=10, pady=(14, 14))

    # ---------- eventos ----------
    def _trocar_modelo(self):
        self.frame_persp.pack_forget()
        self.frame_orto.pack_forget()
        (self.frame_persp if self.modelo.get() == PERSPECTIVA else self.frame_orto).pack(fill="x")
        self._atualizar()

    def _ao_igualar(self):
        if self.igualar.get():
            self.vars["fy"].set(self.vars["fx"].get())
        self._atualizar()

    def _ao_alterar(self, chave):
        if self._atualizando:
            return
        if self.igualar.get() and chave in ("fx", "fy"):
            outro = "fy" if chave == "fx" else "fx"
            self._atualizando = True
            self.vars[outro].set(self.vars[chave].get())
            self._atualizando = False
        self._atualizar()

    def restaurar(self):
        self._atualizando = True
        valores = {**fi.PADRAO_PERSPECTIVA, **fi.PADRAO_EXTRINSECOS, "escala": fi.PADRAO_ESCALA}
        for chave, v in valores.items():
            self.vars[chave].set(f"{v:g}")
        self._atualizando = False
        self.modelo.set(PERSPECTIVA)
        self.resolucao.set("640 x 480")
        self.igualar.select()
        self._trocar_modelo()

    # ---------- interação com o mouse na imagem ----------
    # esquerdo: arrastar gira (X/Y), Shift+arrastar gira em Z | direito: arrasta transladando X/Y
    # roda: aproxima/afasta (tz)
    def _mouse_inicio(self, e):
        self._arrasto = (e.x, e.y)

    def _mouse_fim(self, _e):
        self._arrasto = None

    def _delta(self, e):
        if self._arrasto is None:
            return None
        dx, dy = (e.x - self._arrasto[0]) / self._esc, (e.y - self._arrasto[1]) / self._esc
        self._arrasto = (e.x, e.y)
        return dx, dy

    def _somar(self, **incrementos):
        """Soma incrementos aos parâmetros e atualiza a imagem uma única vez."""
        self._atualizando = True
        try:
            for chave, inc in incrementos.items():
                v = self._valor(chave) + inc
                if chave in ("rx", "ry", "rz"):
                    v = (v + 180) % 360 - 180
                self.vars[chave].set(f"{round(v, 3):g}")
        except ValueError:
            pass
        finally:
            self._atualizando = False
        self._atualizar()

    def _mouse_gira(self, e):
        d = self._delta(e)
        if d is None:
            return
        k = 0.4  # graus por pixel
        if e.state & 0x1:  # Shift
            self._somar(rz=d[0] * k)
        else:
            self._somar(ry=-d[0] * k, rx=d[1] * k)

    def _mouse_translada(self, e):
        d = self._delta(e)
        if d is None:
            return
        try:
            if self.modelo.get() == PERSPECTIVA:
                m = self._valor("tz") / self._valor("fx")   # unidades do mundo por pixel na profundidade tz
            else:
                m = 1 / self._valor("escala")
        except (ValueError, ZeroDivisionError):
            return
        self._somar(tx=d[0] * m, ty=d[1] * m)

    def _mouse_roda(self, e):
        self._somar(tz=-0.5 if e.delta > 0 else 0.5)

    @staticmethod
    def _numero(texto):
        """Converte o texto em float finito; levanta ValueError para vazio, texto, nan ou inf."""
        v = float(texto.replace(",", "."))
        if not math.isfinite(v):
            raise ValueError("valor não finito")
        return v

    def _valor(self, chave):
        return self._numero(self.vars[chave].get())

    # ---------- geração da imagem ----------
    def _atualizar(self):
        if not hasattr(self, "frame_orto"):
            return
        persp = self.modelo.get() == PERSPECTIVA
        try:
            ext = {k: self._valor(k) for k in fi.PADRAO_EXTRINSECOS}
            if persp:
                intr = {k: self._valor(k) for k in fi.PADRAO_PERSPECTIVA}
                K = fi.matriz_intrinseca(**intr)
                self.k_label.configure(text=np.array2string(K, precision=2, suppress_small=True))
                escala = None
            else:
                intr, escala = None, self._valor("escala")
        except ValueError:
            self.status.configure(text="Valor inválido: use apenas números.")
            return
        res = fi.RESOLUCOES[self.resolucao.get()]
        v2d, segs = fi.projetar_cubo("perspectiva" if persp else "ortografica", res, ext, intr,
                                     escala if escala is not None else fi.PADRAO_ESCALA)
        self._ultima = fi.renderizar(v2d, segs, res)
        self.status.configure(text=f"{self.modelo.get()}  |  sensor {res[0]}x{res[1]} px")
        self._exibir()

    def _exibir(self):
        if self._ultima is None:
            return
        cw, ch = max(self.image_label.winfo_width(), 1), max(self.image_label.winfo_height(), 1)
        w, h = self._ultima.size
        esc = self._esc = min(cw / w, ch / h, 1.0)
        tam = (max(int(w * esc), 1), max(int(h * esc), 1))
        self._ctk_image = ctk.CTkImage(light_image=self._ultima, dark_image=self._ultima, size=tam)
        self.image_label.configure(image=self._ctk_image)
