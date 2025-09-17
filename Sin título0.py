# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 19:15:45 2025

@author: Carola
"""
import numpy as np
import matplotlib.pyplot as plt


N = 45 
nfft = 1000 #cantidad de muestras de la transformada

#ventanas sugeridas por el libro
rectangular = np.ones(N)
hamming = np.hamming(N)
hann = np.hanning(N)
blackman = np.blackman(N)

ventanas = {
    "Rectangular": rectangular,
    "Hamming": hamming,
    "Hann": hann,
    "Blackman": blackman
}


w = np.linspace(-np.pi, np.pi, nfft)

plt.figure(figsize=(10,6))


for nombre, ventana in ventanas.items():
    #FFT centrada !!
    W = np.fft.fftshift(np.fft.fft(ventana, nfft))
    W = 20*np.log10(np.abs(W)/np.max(np.abs(W)))  #normalización en dB
    plt.plot(w, W, label=nombre)

plt.ylim([-80, 5])
plt.xlim([-np.pi, np.pi])
plt.ylabel("Magnitud [dB]")
plt.xlabel("Frecuencia [rad/muestra]")
plt.title("Respuesta en frecuencia de distintas ventanas")
plt.legend()
plt.grid(True)
plt.show()
