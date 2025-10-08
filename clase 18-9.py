# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 22:32:47 2025

@author: Carola
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal.windows import flattop, hann, hamming, blackman


N_mat = 1000        
fs_mat = 1000       
df_mat = fs_mat / N_mat
amp_0 = np.sqrt(2)
SNR = 10            # dB
R = 200             # realizaciones

nn = np.arange(N_mat)             
ff = np.arange(N_mat) * df_mat    # eje de frecuencias
tt = nn / fs_mat                  # eje temporal

#senoidal NxR
tt_mat = np.tile(tt.reshape(N_mat,1), (1,R))
frec_mat = np.random.uniform (-2, 2)  # frecuencias aleatorias entre 0 y fs/2
frec_mat=np.tile(frec_mat, (N_mat,1))
xx_mat = amp_0 * np.sin(2*np.pi*frec_mat*tt_mat)

#ruido blanco gaussiano
pot_ruido = amp_0**2 / (2 * 10**(SNR/10))
ruido = np.random.normal(0, np.sqrt(pot_ruido), (N_mat,R))
x_mat = xx_mat + ruido

#ventaneo de alti
flattop_win = flattop(N_mat).reshape(-1,1)   
x_mat_vent = x_mat * flattop_win 

   
#hay que repetir para las 4 ventanas sin agregarle una dimensión. armar senos para cada ventana y hacer el arrastre matricial o algo así         
#ojo con la transferencia
#fft
X_mat = (1/N_mat) * fft(x_mat_vent, axis=0)   # cada columna = una realización

plt.figure(figsize=(10,5))
plt.plot(ff, 10*np.log10(2*np.abs(X_mat)**2))  # esto dibuja todas las realizaciones

plt.grid(True)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PDS [dB]")
plt.title("Ventaneo de señal matricial  ruido")
plt.show()

#fetas de la matriz FFT (filas en índices 0, N/2, N/4)
feta0  = np.abs(X_mat[0, :])         
fetaN2 = np.abs(X_mat[N_mat//2, :]) 
fetaN4 = np.abs(X_mat[N_mat//4, :])

#histograma para una determinada ventana


plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(feta0)


plt.subplot(3,1,2)
plt.plot(fetaN2)
plt.title("Nyquist")

plt.subplot(3,1,3)
plt.plot(fetaN4)
plt.title("mitad de banda digital")

plt.tight_layout()
plt.show()
