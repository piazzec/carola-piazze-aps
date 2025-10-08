# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:42:41 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

#seno con ruido para testear


N = 1000
fs = N
ts = 1/fs

SNR = 10        #relación señal/ruido en dB

def sen( N, fs, frec=5, amp=1, fase=0, dc=0):
    n = np.arange(N)
    t = n / fs
    x = dc + amp * np.sin(2*np.pi*frec*t + fase)
    return t, x
amp = np.sqrt(2)

t, x = sen( N, fs, amp=amp)

#calculo de la potencia de ruido en base a la SNR
pot_snr = amp**2 / (2 * 10**(SNR/10))

#ruido blanco gaussiano (media=0, varianza=pot_snr)
ruido = np.random.normal(0, np.sqrt(pot_snr), N)

#le sumo el ruido
x_ruido = x + ruido


plt.figure(figsize=(10,4))
plt.plot(t, x_ruido, label="Señal con ruido")
plt.plot(t, x, label="Señal pura", alpha=0.8)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal senoidal con ruido blanco (SNR = {} dB)".format(SNR))
plt.xlim(0,0.5)
plt.ylim(-2,2)
plt.grid(True)
plt.legend()
plt.show()


#ejemplo 2, ruido en la transformada con la idea de las matrices
#renombro todo para que no explote
N_mat = 1000        
fs_mat = 1000       
df_mat = fs_mat / N_mat
amp_0 = np.sqrt(2)
SNR = 10            #dB
R = 200             #realizaciones


nn = np.arange(N_mat)             
ff = np.arange(N_mat) * df_mat    #vector de frecuencias
tt = nn / fs_mat                  #eje temporal

#pasamos tt a columna (N x 1) y lo repetimos R veces
tt_mat = np.tile(tt.reshape(N,1), (1,R))
#hace lo que dijo mariano pero todo junto, primero el reshape para hacer las columnas 
#y después ya está el tile con el 1, R


# Frecuencia central + variaciones (esto no lo entiendo)
frec_central = (N_mat/4) * df_mat
frec_rand = np.random.uniform(-2, 2, R) * (fs_mat/N_mat)
frec_mat = np.tile(frec_rand, (N_mat,1))

#matriz de señales NxR 
xx_mat = amp_0 * np.sin(2*np.pi*(frec_central + frec_mat)*tt_mat)

#ruido gaussiano NxR con potencia ajustada
pot_ruido = amp_0**2 / (2 * 10**(SNR/10))
ruido = np.random.normal(0, np.sqrt(pot_ruido), (N_mat,R))

#le meto el ruido a la señal
x_mat = xx_mat + ruido

#fft en el tiempo
X_mat = (1/N_mat) * fft(x_mat, axis=0)

#espectro promedio ESTO NO LO ENTIENDO
X_avg = np.mean(np.abs(X_mat)**2, axis=1)

plt.figure(2)
plt.plot(ff, 10*np.log10(2*X_avg), label="Señal + ruido (promedio)")
plt.xlim((0, fs_mat/2))
plt.grid(True)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dB]")
plt.title("Espectro promedio con ruido")
plt.legend()
plt.show()

#Estimador de energía !!!