import cv2
# Cargar dos imágenes
img1 = cv2.imread('./images/color1.jpeg')
img2 = cv2.imread('./images/binaria1.jpeg')

# Asegurarse de que las imágenes tengan el mismo tamaño
img1 = cv2.resize(img1, (300, 300))
img2 = cv2.resize(img2, (300, 300))
# Operaciones lógicas
and_img = cv2.bitwise_and(img1, img2)
or_img = cv2.bitwise_or(img1, img2)
xor_img = cv2.bitwise_xor(img1, img2)
# Mostrar resultados
cv2.imshow('Imagen AND', and_img)
cv2.imshow('Imagen OR', or_img)
cv2.imshow('Imagen XOR', xor_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
