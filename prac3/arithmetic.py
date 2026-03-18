import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('imagen_color.jpg')

# Operaciones aritméticas
suma = cv2.add(img, 50) # Suma un escalar
resta = cv2.subtract(img, 50) # Resta un escalar
multiplicacion = cv2.multiply(img, 1.2) # Multiplica por un escalar

# Mostrar resultados
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen Suma', suma)
cv2.imshow('Imagen Resta', resta)
cv2.imshow('Imagen Multiplicación', multiplicacion)
cv2.waitKey(0)
cv2.destroyAllWindows()
