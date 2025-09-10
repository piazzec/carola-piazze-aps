# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 21:01:23 2025

@author: Carola
"""

#clase 27/8/2025
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Definición de parámetros
N = 1000  # numero de muestras
fs = N    # frec muestreo
df = fs/N # res espectral
ts = 1/fs # tiempo entre muestras

def sen(ff, nn, amp=1, dc=0, ph=0, fs=2):
    """
    ff: frecuencia
    nn: cant muestras
    amp: amplitud claramente
    dc: offset
    ph: fase
    fs: frec muestreo
    
    """
    N=np.arange(nn)
    t=N/fs
    x = dc + amp * np.sin(2 * np.pi * ff * t + ph)  # genero señal
    return t, x


t1, x1 = sen(ff=(N/4)*df, nn=N, fs=fs)
t2, x2 = sen(ff=((N/4)+1)*df, nn=N, fs=fs)
t2, x3 = sen(ff=((N/4)+0.5)*df, nn=N, fs=fs)

#transformada
xx1=fft(x1)
xx1abs=np.abs(xx1)
xx1ang=np.angle(xx1)

xx2=fft(x2)
xx2abs=np.abs(xx2)
xx2ang=np.angle(xx2)

xx3=fft(x3)
xx3abs=np.abs(xx3)
xx3ang=np.angle(xx3)

Ft=np.arange(N)*df

plt.figure(1)
plt.plot(Ft,np.log10(xx1abs)*20,'x',label='x1 abs en db')
plt.plot(Ft,np.log10(xx2abs)*20,'*',label='x2 abs en db')
plt.plot(Ft,np.log10(xx3abs)*20,'*',label='x3 abs en db')
plt.figure(2)
plt.plot(xx1abs, 'x',label='x1 abs')
plt.plot(xx2abs, '*',label='x2 abs')

plt.title('FFT')
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Amplitud [dB]')

plt.legend()

plt.grid()
plt.tight_layout() 
plt.show()
#%matplotlib para ver gráficos en pestaña aparte!!


""" pasaron alternativa por el chat
plt.figure(1)
plt.plot(freqs, X1abs, 'x', label = 'X1abs')
#plt.plot(freqs, np.log10(X1abs)*20, 'x', label = 'X1abs')
plt.plot(freqs, X2abs, 'o', label = 'X2abs')
plt.plot(freqs, X3abs, 'X', label = 'X3abs')
plt.xlim([0,fs/2])
plt.title('FFT')
plt.xlabel('Frecuencia en Hz')
plt.ylabel('Amplitud ')
plt.legend()
# plt.grid()
plt.show()

plt.figure(2)
plt.plot(freqs, np.log10(X1abs)*20, 'x', label = 'X1abs')
plt.plot(freqs, np.log10(X2abs)*20, 'o', label = 'X2abs')
plt.plot(freqs, np.log10(X3abs)*20, 'X', label = 'X3abs')
plt.xlim([0,fs/2])
plt.title('FFT')
plt.xlabel('Frecuencia en Hz')
plt.ylabel('Amplitud ')
plt.legend()
# plt.grid()
plt.show()
"""

