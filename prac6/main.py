"""
noise_image.py
==============
Aplicación para aplicar ruido a imágenes en escala de grises.

Tipos de ruido disponibles:
  - Sal y pimienta
  - Gaussiano
  - Multiplicativo (Speckle)

Uso:
    python noise_image.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tkinter import Tk, filedialog


# ─────────────────────────────────────────────
# 1. CARGA DE IMAGEN
# ─────────────────────────────────────────────

def cargar_imagen() -> np.ndarray | None:
    """
    Abre el explorador de archivos del sistema (Linux / Windows)
    y devuelve la imagen cargada en escala de grises.

    Returns
    -------
    np.ndarray | None
        Imagen en escala de grises o None si el usuario cancela.
    """
    root = Tk()
    root.withdraw()                      # Oculta la ventana principal de Tkinter
    root.attributes("-topmost", True)    # Ventana siempre al frente

    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG",  "*.png"),
            ("BMP",  "*.bmp"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()

    if not ruta:
        print("⚠  No se seleccionó ninguna imagen.")
        return None

    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"✗  No se pudo leer el archivo: {ruta}")
        return None

    print(f"✔  Imagen cargada: {ruta}  |  tamaño: {img.shape[1]}×{img.shape[0]} px")
    return img


# ─────────────────────────────────────────────
# 2. FUNCIONES DE RUIDO
# ─────────────────────────────────────────────

def ruido_sal_pimienta(imagen: np.ndarray, cantidad: float = 0.05) -> np.ndarray:
    """
    Aplica ruido sal y pimienta.

    Parameters
    ----------
    imagen   : imagen en escala de grises (uint8).
    cantidad : proporción de píxeles afectados [0, 1].

    Returns
    -------
    Imagen con ruido (uint8).
    """
    resultado = imagen.copy()
    total_px  = imagen.size
    n_afectados = int(total_px * cantidad)

    # Sal (píxeles blancos)
    filas = np.random.randint(0, imagen.shape[0], n_afectados // 2)
    cols  = np.random.randint(0, imagen.shape[1], n_afectados // 2)
    resultado[filas, cols] = 255

    # Pimienta (píxeles negros)
    filas = np.random.randint(0, imagen.shape[0], n_afectados // 2)
    cols  = np.random.randint(0, imagen.shape[1], n_afectados // 2)
    resultado[filas, cols] = 0

    return resultado


def ruido_gaussiano(imagen: np.ndarray, media: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Aplica ruido gaussiano aditivo.

    Parameters
    ----------
    imagen : imagen en escala de grises (uint8).
    media  : media de la distribución normal.
    sigma  : desviación estándar (controla la intensidad del ruido).

    Returns
    -------
    Imagen con ruido (uint8).
    """
    ruido    = np.random.normal(media, sigma, imagen.shape).astype(np.float32)
    resultado = np.clip(imagen.astype(np.float32) + ruido, 0, 255)
    return resultado.astype(np.uint8)


def ruido_multiplicativo(imagen: np.ndarray, varianza: float = 0.1) -> np.ndarray:
    """
    Aplica ruido multiplicativo (Speckle): I_r = I + I * N(0, var).

    Parameters
    ----------
    imagen   : imagen en escala de grises (uint8).
    varianza : varianza del ruido gaussiano multiplicado.

    Returns
    -------
    Imagen con ruido (uint8).
    """
    ruido     = np.random.normal(0, varianza ** 0.5, imagen.shape).astype(np.float32)
    resultado = imagen.astype(np.float32) * (1 + ruido)
    resultado = np.clip(resultado, 0, 255)
    return resultado.astype(np.uint8)


# ─────────────────────────────────────────────
# 3. VISUALIZACIÓN
# ─────────────────────────────────────────────

def visualizar_resultados(
    original: np.ndarray,
    sp: np.ndarray,
    gauss: np.ndarray,
    mult: np.ndarray,
) -> None:
    """
    Muestra en una sola figura los cuatro resultados:
    imagen original + tres versiones con ruido, cada una
    acompañada de su histograma.

    Parameters
    ----------
    original : imagen original en escala de grises.
    sp       : imagen con ruido sal y pimienta.
    gauss    : imagen con ruido gaussiano.
    mult     : imagen con ruido multiplicativo.
    """
    imagenes = [original, sp, gauss, mult]
    titulos  = [
        "Original (escala de grises)",
        "Ruido Sal y Pimienta",
        "Ruido Gaussiano",
        "Ruido Multiplicativo (Speckle)",
    ]
    colores_hist = ["#4a90d9", "#e05c5c", "#50c878", "#f0a500"]

    # Paleta general
    BG      = "#0f1117"
    PANEL   = "#1a1d27"
    TEXT    = "#e8eaf0"
    ACCENT  = "#5b8dee"

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        "Análisis de Ruido en Imágenes",
        fontsize=20, fontweight="bold",
        color=TEXT, y=0.98,
        fontfamily="DejaVu Sans",
    )

    # Grid: 2 filas × 4 columnas
    #   fila 0 → imágenes
    #   fila 1 → histogramas
    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.04, right=0.97,
        top=0.92, bottom=0.06,
    )

    for col, (img, titulo, color) in enumerate(zip(imagenes, titulos, colores_hist)):

        # ── Imagen ──────────────────────────────
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax_img.set_title(titulo, fontsize=9.5, color=TEXT,
                         fontweight="bold", pad=6)
        ax_img.axis("off")

        # Borde de color distintivo
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        # Estadísticas de la imagen
        media_val = img.mean()
        std_val   = img.std()
        ax_img.set_xlabel(
            f"μ={media_val:.1f}  σ={std_val:.1f}",
            fontsize=7.5, color="#aab0c0",
        )

        # ── Histograma ──────────────────────────
        ax_hist = fig.add_subplot(gs[1, col])
        ax_hist.set_facecolor(PANEL)
        ax_hist.hist(
            img.ravel(), bins=64,
            range=(0, 255),
            color=color, alpha=0.85,
            edgecolor="none",
        )
        ax_hist.set_xlim(0, 255)
        ax_hist.set_title("Histograma", fontsize=8, color="#aab0c0", pad=4)
        ax_hist.tick_params(colors="#aab0c0", labelsize=7)
        ax_hist.set_xlabel("Intensidad", fontsize=7.5, color="#aab0c0")
        ax_hist.set_ylabel("Frecuencia", fontsize=7.5, color="#aab0c0")

        for spine in ax_hist.spines.values():
            spine.set_edgecolor("#2e3244")
        ax_hist.grid(axis="y", color="#2e3244", linewidth=0.6, linestyle="--")

    # Pie de página con parámetros usados
    fig.text(
        0.5, 0.01,
        "Parámetros: Sal & Pimienta=5 %  |  Gaussiano σ=25  |  Multiplicativo var=0.1",
        ha="center", fontsize=8, color="#6b7280",
    )

    plt.show()
    
def redimensionar_imagen(imagen: np.ndarray, min_dim: int = 200) -> np.ndarray:
    """
    Redimensiona la imagen proporcionalmente para que su dimensión
    mínima (alto o ancho) sea igual a min_dim píxeles.

    Parameters
    ----------
    imagen  : imagen en escala de grises (uint8).
    min_dim : tamaño mínimo deseado en píxeles (default: 200).

    Returns
    -------
    Imagen redimensionada (uint8).
    """
    alto, ancho = imagen.shape
    dim_min = min(alto, ancho)

    if dim_min <= min_dim:
        print(f"ℹ  La imagen ya es menor o igual a {min_dim}px en su dimensión mínima. No se redimensiona.")
        return imagen

    escala      = min_dim / dim_min
    nuevo_ancho = int(ancho * escala)
    nuevo_alto  = int(alto  * escala)

    resultado = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    print(f"✔  Redimensionada: {ancho}×{alto} → {nuevo_ancho}×{nuevo_alto} px  (escala: {escala:.4f})")
    return resultado


# ─────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def main() -> None:
    """Orquesta la carga, procesado y visualización."""

    print("=" * 52)
    print("   Aplicación de Ruido a Imágenes con OpenCV")
    print("=" * 52)

    # 1. Cargar imagen
    imagen_gris = cargar_imagen()
    if imagen_gris is None:
        return
    imagen_gris = redimensionar_imagen(imagen_gris, min_dim=200)
    
    # 2. Aplicar ruidos
    print("⚙  Aplicando ruidos...")
    img_sp    = ruido_sal_pimienta(imagen_gris,   cantidad=0.05)
    img_gauss = ruido_gaussiano(imagen_gris,       media=0, sigma=25)
    img_mult  = ruido_multiplicativo(imagen_gris,  varianza=0.1)
    print("✔  Ruidos aplicados correctamente.")

    # 3. Mostrar resultados
    print("📊 Mostrando resultados...")
    visualizar_resultados(imagen_gris, img_sp, img_gauss, img_mult)
    print("✔  Listo.")


if __name__ == "__main__":
    main()
