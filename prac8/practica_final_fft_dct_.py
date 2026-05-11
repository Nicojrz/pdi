#!/usr/bin/env python3
# Práctica mejorada: La FFT y DCT en acción
# Objetivo: Mostrar aplicaciones reales (limpieza de imagen y compresión)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# Utilidades

def load_image(path, size=256):
    try:
        img = Image.open(path).convert('L')
        img = img.resize((size, size))
        arr = np.asarray(img).astype(np.float32)/255.0
        return arr
    except:
        # Imagen sintética con ruido
        x = np.arange(size)
        y = np.arange(size)
        X,Y = np.meshgrid(x,y)
        checker = (((X//16)%2)^((Y//16)%2)).astype(np.float32)
        noise = 0.1*np.random.randn(size,size).astype(np.float32)
        img = np.clip(checker+noise,0,1)
        return img

# FFT y filtrado

def fft_filter(img, cutoff=0.15, tipo='lowpass'):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    h,w = img.shape
    cy,cx = h//2,w//2
    Y,X = np.ogrid[:h,:w]
    dist = np.sqrt((Y-cy)**2+(X-cx)**2)
    mask = (dist<=cutoff*min(h,w)) if tipo=='lowpass' else (dist>cutoff*min(h,w))
    Fshift_filtered = Fshift*mask
    img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift_filtered)))
    img_filtered = np.clip(img_filtered,0,1)
    return img_filtered, mask

# DCT compresión

def dct_matrix(n=8):
    A = np.zeros((n,n))
    for k in range(n):
        alpha = math.sqrt(1/n) if k==0 else math.sqrt(2/n)
        for x in range(n):
            A[k,x]=alpha*math.cos(((2*x+1)*k*math.pi)/(2*n))
    return A

def dct_compress(img,q_factor=0.5):
    img255=(img*255)-128
    h,w=img255.shape
    A=dct_matrix(8)
    out=np.zeros_like(img255)
    Q=np.ones((8,8))*16*q_factor
    for i in range(0,h,8):
        for j in range(0,w,8):
            block=img255[i:i+8,j:j+8]
            C=A@block@A.T
            Cq=np.round(C/Q)*Q
            rec=A.T@Cq@A
            out[i:i+8,j:j+8]=rec
    out=(out+128)/255
    out=np.clip(out,0,1)
    return out

def psnr(a,b):
    mse=np.mean((a-b)**2)
    return 20*math.log10(1/math.sqrt(mse)) if mse>0 else float('inf')

# Visualización

def show_results(original,filtered,compressed,q):
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(original,cmap='gray');axes[0].set_title('Original')
    axes[1].imshow(filtered,cmap='gray');axes[1].set_title('Filtrada (FFT)')
    axes[2].imshow(compressed,cmap='gray');axes[2].set_title(f'Comprimida DCT (q={q})')
    for ax in axes: ax.axis('off')
    plt.tight_layout();plt.show()
    print(f"PSNR compresión: {psnr(original,compressed):.2f} dB")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--imagen',type=str,default='data/ejemplo.png')
    parser.add_argument('--cutoff',type=float,default=0.15)
    parser.add_argument('--tipo',type=str,default='lowpass')
    parser.add_argument('--dct_q',type=float,default=0.5)
    args=parser.parse_args()

    img=load_image(args.imagen)
    filtered,_=fft_filter(img,args.cutoff,args.tipo)
    compressed=dct_compress(img,args.dct_q)
    show_results(img,filtered,compressed,args.dct_q)
