import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def entropia_kapur(histograma,total_pixeles):
    max_entropia = -1
    umbral_optimo = 0
    for t in range(1,255):
        clase1 = histograma[:t]
        clase2 = histograma[t:]
        p1 = np.sum(clase1) / total_pixeles
        p2 = np.sum(clase2) / total_pixeles
        if p1 == 0 or p2 == 0:
            continue
        entropia1 = -np.sum((clase1 / np.sum(clase1)) * np.log(clase1 / np.sum(clase1) + 1e-10))
        entropia2 = -np.sum((clase2 / np.sum(clase2)) * np.log(clase2 / np.sum(clase2) + 1e-10))
        entropia_total = p1 * entropia1 + p2 * entropia2
        if entropia_total > max_entropia:
            max_entropia = entropia_total
            umbral_optimo = t
    return umbral_optimo

image = cv2.imread('cartas.jpg')

col = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gris, (5,5), 0)

_, binary_image = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(col)
plt.title('Imagen original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(gris, cmap='gray')
plt.title('Imagen en grises')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Imagen Binarizada')
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(gris.ravel(), bins=256, range=[0, 256])
plt.title('Histograma de la imagen')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Frecuencia')
plt.show()

_, umbral_otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

histograma, _ = np.histogram(gris, bins=256, range=(0, 256))
total_pixeles = gris.size
umbral_kapur = entropia_kapur(histograma, total_pixeles)
imagen_kapur = (gris > umbral_kapur).astype(np.uint8) * 255
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(umbral_otsu, cmap='gray')
plt.title('Segmentación - Método de Otsu')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(imagen_kapur, cmap='gray')
plt.title(f'Segmentación - Método de Kapur (Umbral: {umbral_kapur})')
plt.axis('off')
plt.show()

picos, _ = find_peaks(histograma, distance=20)
minimo = np.argmin(histograma[picos[0]:picos[1]]) + picos[0]
imagen_minimo = (gris > minimo).astype(np.uint8) * 255
plt.figure(figsize=(5, 5))
plt.imshow(imagen_minimo, cmap='gray')
plt.title(f'Segmentación - Mínimo del Histograma (Umbral: {minimo})')
plt.axis('off')
plt.show()

umbral_media = np.mean(gris)
imagen_segmentada_media = (gris >= umbral_media).astype(np.uint8) * 255
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gris, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(imagen_segmentada_media, cmap='gray')
plt.title(f'Segmentación - Umbral Media ({umbral_media:.2f})')
plt.axis('off')
plt.show()

T1 = 80
T2 = 150
imagen_multi_umbrales = np.zeros_like(gris)
imagen_multi_umbrales[gris < T1] = 0
imagen_multi_umbrales[(gris >= T1) & (gris < T2)] = 127
imagen_multi_umbrales[gris >= T2] = 255
imagen_umbral_banda = np.zeros_like(gris)
imagen_umbral_banda[(gris >= T1) & (gris <= T2)] = 255
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(gris, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(imagen_multi_umbrales, cmap='gray')
plt.title(f'Múltiples Umbrales: T1={T1}, T2={T2}')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(imagen_umbral_banda, cmap='gray')
plt.title(f'Umbral Banda: [{T1}, {T2}]')
plt.axis('off')
plt.tight_layout()
plt.show()

uniforme = cv2.equalizeHist(gris)

exp = np.uint8(255 * (1 - np.exp(-gris / 255)))

rayleigh = np.uint8(255 * np.sqrt(gris / 255))
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(uniforme, cmap='gray')
plt.title('Ecualización Uniforme')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(exp, cmap='gray')
plt.title('Ecualización Exponencial')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(rayleigh, cmap='gray')
plt.title('Ecualización Rayleigh')
plt.axis('off')
plt.show()

hipercubica = np.uint8(255 * (gris / 255) ** 4)

log_hiper = np.uint8(255 * np.log1p(gris) / np.log1p(255))

potencia = np.uint8(255 * (gris / 255) ** 2)

gamma = 1.5
imagen_gamma = np.power(gris / 255.0, gamma) * 255
imagen_gamma = np.uint8(imagen_gamma)
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(hipercubica, cmap='gray')
plt.title('Ecualización Hipercúbica')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(log_hiper, cmap='gray')
plt.title('    Ecualización Logarítmica Hiperbólica')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(potencia, cmap='gray')
plt.title('Función Potencia')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(imagen_gamma, cmap='gray')
plt.title(f'Corrección Gamma (γ={gamma})')
plt.axis('off')
plt.show()

segmentacion=umbral_otsu
mascara=segmentacion.copy()

image_color = cv2.cvtColor(segmentacion, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(segmentacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    cv2.drawContours(image_color, [contour], -1, (0, 255, 0), 2)
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Objetos Detectados y Numerados')
plt.axis('off')
plt.show()


resultado=cv2.bitwise_and(col,col,mask=mascara)
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(col)
plt.title("Imagen Original")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(mascara, cmap='gray')
plt.title("Máscara")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(resultado)
plt.title("Segmentación Aplicada")
plt.axis("off")
plt.show()