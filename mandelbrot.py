import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time

def mandelbrot(height, width, n, lims=(-2., 2., -2., 2.)):
    X, Y = np.meshgrid(np.linspace(lims[0], lims[1], width), np.linspace(lims[3], lims[2], height))
    # z = a+bi
    a = np.zeros((height, width), dtype=np.float64)
    b = np.zeros_like(a)
    escape_time = np.zeros((height, width), dtype=np.int32)
    for i in range(1, n+1):
        not_escaped = a*a+b*b < 4
        atemp = a*a - b*b + X
        b[not_escaped] = (2*a*b + Y)[not_escaped]
        a[not_escaped] = atemp[not_escaped]
        mask = ~not_escaped & (escape_time == 0)
        escape_time[mask] = i
    return escape_time

if __name__=='__main__':
    # lims = (-0.4, 0.1, 0.6, 1.1)
    lims = (-2, 2, -2, 2)
    st = time()
    out = mandelbrot(1000, 1000, 256, lims)
    en = time()
    print(f"Time to run: {en-st:.3f} s")
    plt.imshow(out, extent=lims, cmap='plasma')
    plt.show()