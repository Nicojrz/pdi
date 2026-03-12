import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Cargar imagen y convertir a escala de grises
imagen = cv2.imread('retrato.jpg')
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Definir mapas
colores_pastel = [
    (1.0, 0.8, 0.9),
    (0.8, 1.0, 0.8),
    (0.8, 0.9, 1.0),
    (1.0, 1.0, 0.8),
    (0.9, 0.8, 1.0)
]
colores_tierra = [
    (0.6, 0.4, 0.2),
    (0.8, 0.7, 0.5),
    (0.9, 0.8, 0.6),
    (0.7, 0.5, 0.3),
    (0.5, 0.3, 0.1)
]
colores_arcoiris = [
    (0.5, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.6, 1.0),
    (0.0, 0.9, 0.4),
    (1.0, 1.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 0.0, 0.0)
]

mapa_pastel   = LinearSegmentedColormap.from_list("PastelMap", colores_pastel,   N=256)
mapa_tierra   = LinearSegmentedColormap.from_list("Tierra",    colores_tierra,   N=256)
mapa_arcoiris = LinearSegmentedColormap.from_list("Arcoiris",  colores_arcoiris, N=256)

# Visualizar
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(imagen_gris, cmap='gray')
axs[0].set_title('Escala de grises')
axs[0].axis('off')

axs[1].imshow(imagen_gris, cmap=mapa_pastel)
axs[1].set_title('Pastel')
axs[1].axis('off')

axs[2].imshow(imagen_gris, cmap=mapa_tierra)
axs[2].set_title('Tierra')
axs[2].axis('off')

axs[3].imshow(imagen_gris, cmap=mapa_arcoiris)
axs[3].set_title('Arcoiris')
axs[3].axis('off')

plt.tight_layout()
plt.show()