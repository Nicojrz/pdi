import cv2
import matplotlib.pyplot as plt
# Cargar una imagen en escala de grises
imagen_gris = cv2.imread('imagen_gris.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar diferentes mapas de color (pseudocolor)
imagen_jet = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_JET)
imagen_hot = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_BONE)
imagen_ocean = cv2.applyColorMap(imagen_gris, cv2.COLORMAP_PINK)

# Mostrar las imágenes en una cuadrícula para comparación visual
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(imagen_gris, cmap='gray')
axs[0, 0].set_title('Imagen en escala de grises')
axs[0, 1].imshow(cv2.cvtColor(imagen_jet, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Pseudocolor: JET')
axs[1, 0].imshow(cv2.cvtColor(imagen_hot, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Pseudocolor: HOT')
axs[1, 1].imshow(cv2.cvtColor(imagen_ocean, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Pseudocolor: OCEAN')

# Quitar los ejes para mejor visualización
for ax in axs.flat:
    ax.axis('off')

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
