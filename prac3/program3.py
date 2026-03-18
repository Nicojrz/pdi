import cv2
import numpy as np

def loadimg(location):
    return cv2.imread(location)

def squarize(img, px):
    return cv2.resize(img, (px, px))

def showimg(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def askimg():
    name = input("Ingresa un nombre para la imagen: ")
    location = input("Ingresa la ruta de la imagen: ")
    px = int(input("Ingresa el tamaño de la imagen: "))
    img = loadimg(location)
    img = squarize(img, px)
    return name, img

if __name__ == "__main__":
    name, img = askimg()
    showimg(name, img)
    cv2.destroyAllWindows()
