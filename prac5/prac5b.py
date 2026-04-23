import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# =========================
# Utilidades de visualización (DRY)
# =========================
def show_image(img, title='', cmap='gray', size=(5, 5)):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_histogram(img):
    plt.figure(figsize=(8, 4))
    plt.hist(img.ravel(), bins=256, range=[0, 256])
    plt.title('Histograma')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.show()


# =========================
# Preprocesamiento
# =========================
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return image


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, ksize=(5, 5)):
    return cv2.GaussianBlur(gray, ksize, 0)


# =========================
# Métodos de umbralización
# =========================
def threshold_fixed(img, t=127):
    _, binary = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(img):
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def kapur_entropy_threshold(hist, total):
    eps = 1e-12
    max_entropy = -np.inf
    best_t = 0

    for t in range(1, 255):
        h1 = hist[:t]
        h2 = hist[t:]

        p1 = np.sum(h1) / total
        p2 = np.sum(h2) / total
        if p1 < eps or p2 < eps:
            continue

        p1_norm = h1 / (np.sum(h1) + eps)
        p2_norm = h2 / (np.sum(h2) + eps)

        e1 = -np.sum(p1_norm * np.log(p1_norm + eps))
        e2 = -np.sum(p2_norm * np.log(p2_norm + eps))

        entropy = p1 * e1 + p2 * e2

        if entropy > max_entropy:
            max_entropy = entropy
            best_t = t

    return best_t


def threshold_kapur(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    t = kapur_entropy_threshold(hist, gray.size)
    return (gray > t).astype(np.uint8) * 255, t


def threshold_histogram_minimum(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    peaks, _ = find_peaks(hist, distance=20)

    if len(peaks) < 2:
        return threshold_otsu(gray), None

    minimum = np.argmin(hist[peaks[0]:peaks[1]]) + peaks[0]
    return (gray > minimum).astype(np.uint8) * 255, minimum


def threshold_mean(gray):
    t = np.mean(gray)
    return (gray >= t).astype(np.uint8) * 255, t


def multi_threshold(gray, T1=28, T2=100):
    img = np.zeros_like(gray)
    img[gray < T1] = 255
    img[(gray >= T1) & (gray < T2)] = 127
    img[gray >= T2] = 255
    return img


def band_threshold(gray, T1=80, T2=150):
    img = np.zeros_like(gray)
    img[(gray >= T1) & (gray <= T2)] = 255
    return img


# =========================
# Ecualizaciones y transformaciones
# =========================
def equalizations(gray):
    return {
        "Rayleigh": np.uint8(255 * np.sqrt(gray / 255))
    }
    """
    return {
        "Uniforme": cv2.equalizeHist(gray),
        "Exponencial": np.uint8(255 * (1 - np.exp(-gray / 255))),
        "Rayleigh": np.uint8(255 * np.sqrt(gray / 255)),
        "Hipercubica": np.uint8(255 * (gray / 255) ** 4),
        "Logaritmica": np.uint8(255 * np.log1p(gray) / np.log1p(255)),
        "Potencia": np.uint8(255 * (gray / 255) ** 2),
    }
    """


def gamma_correction(gray, gamma=1.5):
    img = np.power(gray / 255.0, gamma) * 255
    return np.uint8(img)

def invert_image(gray):
    """Invierte intensidades: I' = 255 - I"""
    return 255 - gray
    
def max_filter(gray, ksize=3):
    """
    Filtro de máximos (dilatación en grises)
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(gray, kernel)


def min_filter(gray, ksize=3):
    """
    Filtro de mínimos (erosión en grises)
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(gray, kernel)
    
def mask(original_bgr, mask_binary):
    """
    Aplica una máscara binaria (0/255) a la imagen original en BGR.
    Conserva solo las zonas donde la máscara es blanca.
    """
    if len(mask_binary.shape) != 2:
        raise ValueError("La máscara debe ser de un solo canal (binaria en grises)")

    # Asegurar que la máscara sea 0/255 uint8
    mask = (mask_binary > 0).astype(np.uint8) * 255

    # Aplicar máscara
    result = cv2.bitwise_and(original_bgr, original_bgr, mask=mask)
    return result

# =========================
# Pipeline principal (SRP)
# =========================
def process_image(path):
    image = load_image(path)
    show_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'Imagen original', cmap=None)

    gray = to_gray(image)
    """
    show_image(gray, 'Grises')
    show_histogram(gray)

    blur = gaussian_blur(gray)

    show_image(threshold_fixed(blur), 'Umbral Fijo')
    show_image(threshold_otsu(gray), 'Otsu')

    kapur_img, t_kapur = threshold_kapur(gray)
    show_image(kapur_img, f'Kapur (t={t_kapur})')

    min_img, t_min = threshold_histogram_minimum(gray)
    show_image(min_img, f'Mínimo Histograma (t={t_min})')

    mean_img, t_mean = threshold_mean(gray)
    show_image(mean_img, f'Media (t={t_mean:.2f})')
    """

    """
    show_image(multi_threshold(gray), 'Múltiples Umbrales')
    show_image(band_threshold(gray), 'Umbral Banda')

    """
    for name, img in equalizations(gray).items():
        show_image(img, f'Ecualización {name}')
        show_histogram(img)
        img = multi_threshold(img, 87, 155)
        show_image(img, 'Múltiples Umbrales + Rayleigh')
        img = max_filter(img)
        show_image(img, 'Filtro Máximos')
        img = invert_image(img)
        show_image(img, 'Imagen invertida')
        img = max_filter(img)
        #img = gaussian_blur(img)
        #img = threshold_fixed(img,140)
        show_image(img, 'Filtro Máximos')
        #img = max_filter(img)
        #img = max_filter(img)
        img = mask(image, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show_image(img, 'Mascara')
        show_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'Imagen original', cmap=None)

#    show_image(gamma_correction(gray), 'Corrección Gamma')

# =========================
# Ejecutar
# =========================
if __name__ == "__main__":
    process_image("cartas.jpg")