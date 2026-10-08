"""Atividade Prática 00 - Interface Gráfica (casca para algoritmos de visão computacional)."""
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

import exercicios

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Visão Computacional")
        self.geometry("1000x700")
        self.minsize(700, 500)

        self.original = None   # imagem original (PIL), nunca modificada
        self.current = None    # imagem atual (PIL), alvo dos algoritmos
        self._ctk_image = None

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        tab_img = self.tabs.add("Imagem")
        tab_ex = self.tabs.add("Exercícios")

        tab_img.grid_columnconfigure(1, weight=1)
        tab_img.grid_rowconfigure(0, weight=1)
        self._build_sidebar(tab_img)

        self.image_label = ctk.CTkLabel(tab_img, text="Abra uma imagem para começar", font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_label.bind("<Configure>", lambda e: self.show())

        self.status = ctk.CTkLabel(tab_img, text="Nenhuma imagem carregada.", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

        self._build_exercises(tab_ex)

        self.bind("<Control-o>", lambda e: self.load_image())
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-r>", lambda e: self.restore_image())

    # ---------- construção da interface ----------
    def _build_sidebar(self, parent):
        side = ctk.CTkFrame(parent, width=210, corner_radius=0)
        side.grid(row=0, column=0, rowspan=2, sticky="nsw")
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="Visão Computacional", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 15), padx=15)

        ctk.CTkLabel(side, text="ARQUIVO", anchor="w", text_color="gray").pack(fill="x", padx=15)
        ctk.CTkButton(side, text="Abrir imagem", command=self.load_image).pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(side, text="Salvar imagem", command=self.save_image).pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(side, text="OPERAÇÕES", anchor="w", text_color="gray").pack(fill="x", padx=15, pady=(20, 0))
        ctk.CTkButton(side, text="Zerar intensidade", command=self.zero_image).pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(side, text="Restaurar original", fg_color="gray30", hover_color="gray40",
                      command=self.restore_image).pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(side, text="TEMA", anchor="w", text_color="gray").pack(fill="x", padx=15, pady=(20, 0))
        ctk.CTkOptionMenu(side, values=["Dark", "Light", "System"],
                          command=lambda m: ctk.set_appearance_mode(m)).pack(fill="x", padx=15, pady=5)

    def _build_exercises(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        lista = ctk.CTkScrollableFrame(parent, width=290)
        lista.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=4)
        slide_atual = None
        for i, (slide, titulo, _, _) in enumerate(exercicios.EXERCICIOS):
            if slide != slide_atual:
                slide_atual = slide
                ctk.CTkLabel(lista, text=slide.upper(), anchor="w", text_color="gray").pack(fill="x", pady=(10, 2))
            ctk.CTkButton(lista, text=titulo, anchor="w", command=lambda i=i: self.run_exercise(i)).pack(fill="x", pady=2)

        direita = ctk.CTkFrame(parent, fg_color="transparent")
        direita.grid(row=0, column=1, sticky="nsew")
        direita.grid_columnconfigure(0, weight=1)
        direita.grid_rowconfigure(1, weight=1)
        ctk.CTkButton(direita, text="Resolver todos", width=140, command=self.run_all).grid(row=0, column=0, sticky="w", pady=(4, 6))
        self.output = ctk.CTkTextbox(direita, font=ctk.CTkFont(family="Consolas", size=13), wrap="none")
        self.output.grid(row=1, column=0, sticky="nsew")
        self.output.insert("end", "Selecione um exercício na lista ao lado.")

    def _write(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)

    def run_exercise(self, i):
        try:
            self._write(exercicios.resolver(i))
        except Exception as exc:
            self._write(f"Erro ao resolver: {exc}")

    def run_all(self):
        partes = []
        for i in range(len(exercicios.EXERCICIOS)):
            try:
                partes.append(exercicios.resolver(i))
            except Exception as exc:
                partes.append(f"Exercício {i + 1}: erro {exc}\n")
        self._write(("=" * 60 + "\n").join(partes))

    # ---------- funcionalidades ----------
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
