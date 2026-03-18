import cv2

img = cv2.imread('./images/binaria2.jpeg', cv2.IMREAD_GRAYSCALE)

# Otsu
_, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Binaria Otsu', binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()
