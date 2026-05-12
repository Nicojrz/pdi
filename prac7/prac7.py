"""
Práctica 7 - Morfología Matemática Binaria y en Lattice
Profa. María Elena Cruz Meza | ESCOM IPN
Herramienta interactiva con GUI usando Tkinter + OpenCV + Matplotlib
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# ─────────────────────────────────────────────
#  Estado global
# ─────────────────────────────────────────────
state = {
    "original": None,
    "current":  None,
    "filepath": None,
}

# ─────────────────────────────────────────────
#  Operaciones
# ─────────────────────────────────────────────

def get_kernel():
    size = int(kernel_size_var.get())
    shape_name = kernel_shape_var.get()
    shape_map = {
        "Rectangular": cv2.MORPH_RECT,
        "Elíptico":    cv2.MORPH_ELLIPSE,
        "Cruz":        cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shape_map[shape_name], (size, size))


def apply_noise():
    if state["original"] is None:
        messagebox.showwarning("Sin imagen", "Primero carga una imagen.")
        return
    img = state["original"].copy()
    noise_type = noise_var.get()
    amount = noise_amount_var.get() / 100.0  # 0-1

    if noise_type == "Unipolar (Sal)":
        mask = np.random.rand(*img.shape[:2]) < amount
        img[mask] = 255

    elif noise_type == "Unipolar (Pimienta)":
        mask = np.random.rand(*img.shape[:2]) < amount
        img[mask] = 0

    elif noise_type == "Bipolar (Sal y Pimienta)":
        half = amount / 2
        salt   = np.random.rand(*img.shape[:2]) < half
        pepper = np.random.rand(*img.shape[:2]) < half
        img[salt]   = 255
        img[pepper] = 0

    elif noise_type == "Gaussiano":
        sigma = noise_amount_var.get()   # reusamos el slider como sigma
        gauss = np.random.normal(0, sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

    state["current"] = img
    show_images(state["original"], img,
                "Original", f"Ruido: {noise_type}  ({int(noise_amount_var.get())})")


def apply_erosion():
    _apply_morph("Erosión", cv2.erode)

def apply_dilation():
    _apply_morph("Dilatación", cv2.dilate)

def apply_opening():
    if state["current"] is None:
        messagebox.showwarning("Sin imagen", "Carga o modifica una imagen primero.")
        return
    k   = get_kernel()
    img = state["current"]
    iters = int(iter_var.get())
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=iters)
    state["current"] = result
    show_images(state["original"], result, "Original", f"Apertura (iter={iters})")

def apply_closing():
    if state["current"] is None:
        messagebox.showwarning("Sin imagen", "Carga o modifica una imagen primero.")
        return
    k   = get_kernel()
    img = state["current"]
    iters = int(iter_var.get())
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=iters)
    state["current"] = result
    show_images(state["original"], result, "Original", f"Cierre (iter={iters})")

def _apply_morph(name, func):
    if state["current"] is None:
        messagebox.showwarning("Sin imagen", "Carga o modifica una imagen primero.")
        return
    k   = get_kernel()
    iters = int(iter_var.get())
    result = func(state["current"], k, iterations=iters)
    state["current"] = result
    show_images(state["original"], result, "Original", f"{name} (iter={iters})")

def reset_image():
    if state["original"] is None:
        return
    state["current"] = state["original"].copy()
    show_images(state["original"], state["current"], "Original", "Imagen restaurada")

def save_image():
    if state["current"] is None:
        messagebox.showwarning("Sin imagen", "No hay imagen procesada para guardar.")
        return
    base, ext = os.path.splitext(state["filepath"])
    out_path = base + "_morfologia" + ext
    cv2.imwrite(out_path, state["current"])
    messagebox.showinfo("Guardada", f"Imagen guardada en:\n{out_path}")

# ─────────────────────────────────────────────
#  Visualización
# ─────────────────────────────────────────────

def show_images(img1, img2, title1="Original", title2="Procesada"):
    ax_left.clear()
    ax_right.clear()

    cmap = "gray"
    ax_left.imshow(img1,  cmap=cmap)
    ax_right.imshow(img2, cmap=cmap)

    ax_left.set_title(title1,  color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    ax_right.set_title(title2, color="#7dd3fc", fontsize=11, fontweight="bold", pad=8)

    for ax in (ax_left, ax_right):
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=1.5)
    canvas.draw()

def load_image():
    path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")]
    )
    if not path:
        return
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        return
    state["original"] = img
    state["current"]  = img.copy()
    state["filepath"] = path
    filename_label.config(text=os.path.basename(path))
    show_images(img, img, "Original", "Original")

# ─────────────────────────────────────────────
#  Construcción de la GUI
# ─────────────────────────────────────────────

root = tk.Tk()
root.title("Morfología Matemática — Práctica 7 | ESCOM IPN")
root.configure(bg="#0f172a")
root.geometry("1300x820")
root.resizable(True, True)

# ── Paleta de colores ──────────────────────────────────
BG      = "#0f172a"
PANEL   = "#1e293b"
ACCENT  = "#0ea5e9"
ACCENT2 = "#38bdf8"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
BTN_BG  = "#1e3a5f"
BTN_FG  = "#7dd3fc"
BTN_ACT = "#0369a1"
DANGER  = "#be185d"
SUCCESS = "#065f46"

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_LABEL  = ("Courier New", 10)
FONT_BTN    = ("Courier New", 10, "bold")
FONT_SMALL  = ("Courier New", 9)

# ── Layout principal ───────────────────────────────────
main_frame = tk.Frame(root, bg=BG)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Columna izquierda: controles
ctrl_frame = tk.Frame(main_frame, bg=PANEL, bd=0, relief="flat", width=280)
ctrl_frame.pack(side="left", fill="y", padx=(0, 10))
ctrl_frame.pack_propagate(False)

# Columna derecha: canvas matplotlib
canvas_frame = tk.Frame(main_frame, bg=BG)
canvas_frame.pack(side="left", fill="both", expand=True)

# ── Título del panel ───────────────────────────────────
title_bar = tk.Frame(ctrl_frame, bg=ACCENT, height=4)
title_bar.pack(fill="x")

tk.Label(ctrl_frame, text="MM · PRÁCTICA 7", bg=PANEL, fg=ACCENT2,
         font=("Courier New", 14, "bold")).pack(pady=(16, 2))
tk.Label(ctrl_frame, text="ESCOM  ·  IPN", bg=PANEL, fg=MUTED,
         font=FONT_SMALL).pack(pady=(0, 14))

sep = tk.Frame(ctrl_frame, bg=ACCENT, height=1)
sep.pack(fill="x", padx=16, pady=(0, 14))

# ── Sección: Cargar imagen ─────────────────────────────
def section_label(parent, text):
    tk.Label(parent, text=text, bg=PANEL, fg=MUTED,
             font=FONT_SMALL, anchor="w").pack(fill="x", padx=16, pady=(10, 2))

def action_btn(parent, text, cmd, color=BTN_BG, fg=BTN_FG):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=color, fg=fg, activebackground=BTN_ACT, activeforeground="white",
                  font=FONT_BTN, relief="flat", cursor="hand2",
                  pady=7, padx=10, bd=0)
    b.pack(fill="x", padx=16, pady=3)
    return b

section_label(ctrl_frame, "▸ ARCHIVO")
action_btn(ctrl_frame, "📂  Cargar imagen", load_image, color="#0c4a6e", fg="#bae6fd")

filename_label = tk.Label(ctrl_frame, text="(ninguna)", bg=PANEL, fg=MUTED,
                          font=FONT_SMALL, wraplength=220, justify="center")
filename_label.pack(pady=(0, 6))

# ── Sección: Elemento estructurante ───────────────────
section_label(ctrl_frame, "▸ ELEMENTO ESTRUCTURANTE")

tk.Label(ctrl_frame, text="Tamaño (píxeles)", bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor="w", padx=16)
kernel_size_var = tk.StringVar(value="5")
size_frame = tk.Frame(ctrl_frame, bg=PANEL)
size_frame.pack(fill="x", padx=16, pady=(2, 6))
for s in ["3", "5", "7", "9", "11"]:
    rb = tk.Radiobutton(size_frame, text=s, variable=kernel_size_var, value=s,
                        bg=PANEL, fg=TEXT, selectcolor=ACCENT, activebackground=PANEL,
                        font=FONT_SMALL)
    rb.pack(side="left", padx=2)

tk.Label(ctrl_frame, text="Forma", bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor="w", padx=16)
kernel_shape_var = tk.StringVar(value="Rectangular")
shape_menu = ttk.Combobox(ctrl_frame, textvariable=kernel_shape_var,
                           values=["Rectangular", "Elíptico", "Cruz"],
                           state="readonly", font=FONT_SMALL, width=20)
shape_menu.pack(padx=16, pady=(2, 6))

tk.Label(ctrl_frame, text="Iteraciones", bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor="w", padx=16)
iter_var = tk.StringVar(value="1")
iter_spin = tk.Spinbox(ctrl_frame, from_=1, to=10, textvariable=iter_var,
                       bg=PANEL, fg=TEXT, insertbackground=TEXT,
                       font=FONT_SMALL, width=5, relief="flat", bd=1)
iter_spin.pack(anchor="w", padx=16, pady=(2, 10))

# ── Sección: Ruido ─────────────────────────────────────
section_label(ctrl_frame, "▸ RUIDO")

noise_var = tk.StringVar(value="Bipolar (Sal y Pimienta)")
noise_menu = ttk.Combobox(ctrl_frame, textvariable=noise_var,
                           values=["Unipolar (Sal)",
                                   "Unipolar (Pimienta)",
                                   "Bipolar (Sal y Pimienta)",
                                   "Gaussiano"],
                           state="readonly", font=FONT_SMALL, width=24)
noise_menu.pack(padx=16, pady=(2, 4))

noise_amount_var = tk.DoubleVar(value=5)
tk.Label(ctrl_frame, text="Intensidad / Sigma", bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor="w", padx=16)
noise_slider = tk.Scale(ctrl_frame, from_=1, to=50, variable=noise_amount_var,
                        orient="horizontal", bg=PANEL, fg=TEXT,
                        troughcolor="#334155", highlightthickness=0,
                        sliderlength=16, length=220, font=FONT_SMALL)
noise_slider.pack(padx=16, pady=(0, 4))
action_btn(ctrl_frame, "⚡  Aplicar ruido", apply_noise, color="#4c1d95", fg="#ddd6fe")

# ── Sección: Operaciones morfológicas ─────────────────
section_label(ctrl_frame, "▸ MORFOLOGÍA EN LATTICE")

op_frame = tk.Frame(ctrl_frame, bg=PANEL)
op_frame.pack(fill="x", padx=16, pady=4)

def morph_btn(parent, text, cmd, col, row):
    b = tk.Button(parent, text=text, command=cmd,
                  bg="#1e3a5f", fg=BTN_FG, activebackground=BTN_ACT,
                  activeforeground="white", font=FONT_BTN,
                  relief="flat", cursor="hand2", pady=8, width=11, bd=0)
    b.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
    return b

morph_btn(op_frame, "⊖ Erosión",   apply_erosion,  0, 0)
morph_btn(op_frame, "⊕ Dilatación", apply_dilation, 1, 0)
morph_btn(op_frame, "○ Apertura",  apply_opening,  0, 1)
morph_btn(op_frame, "● Cierre",    apply_closing,  1, 1)
op_frame.columnconfigure(0, weight=1)
op_frame.columnconfigure(1, weight=1)

# ── Botones de utilidad ────────────────────────────────
sep2 = tk.Frame(ctrl_frame, bg=ACCENT, height=1)
sep2.pack(fill="x", padx=16, pady=(14, 8))

action_btn(ctrl_frame, "↺  Restaurar original", reset_image, color="#374151", fg="#d1d5db")
action_btn(ctrl_frame, "💾  Guardar imagen", save_image, color=SUCCESS, fg="#6ee7b7")

# ── Info footer ────────────────────────────────────────
tk.Label(ctrl_frame,
         text="Aplica ruido → morfología\nLa imagen actual se va acumulando",
         bg=PANEL, fg=MUTED, font=FONT_SMALL, justify="center").pack(pady=(12, 4))

# ── Canvas matplotlib ──────────────────────────────────
fig = Figure(figsize=(9.5, 6.5), facecolor="#0f172a")
ax_left  = fig.add_subplot(1, 2, 1)
ax_right = fig.add_subplot(1, 2, 2)
for ax in (ax_left, ax_right):
    ax.set_facecolor("#1e293b")
    ax.axis("off")
fig.text(0.5, 0.97,
         "Morfología Matemática Binaria y en Lattice",
         color="#7dd3fc", fontsize=13, fontweight="bold",
         ha="center", va="top", fontfamily="monospace")

canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Mensaje de bienvenida en el canvas
ax_left.text(0.5, 0.5, "Carga una imagen\npara comenzar",
             color="#475569", fontsize=14, ha="center", va="center",
             fontfamily="monospace", transform=ax_left.transAxes)
ax_right.text(0.5, 0.5, "La imagen procesada\naparecerá aquí",
              color="#475569", fontsize=14, ha="center", va="center",
              fontfamily="monospace", transform=ax_right.transAxes)
canvas.draw()

# ── Estilo ttk ─────────────────────────────────────────
style = ttk.Style()
style.theme_use("clam")
style.configure("TCombobox",
                fieldbackground="#1e3a5f",
                background="#1e3a5f",
                foreground=TEXT,
                selectbackground=ACCENT,
                selectforeground="white")
style.map("TCombobox", fieldbackground=[("readonly", "#1e3a5f")])

# ── Arranque ───────────────────────────────────────────
root.mainloop()