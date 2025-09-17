# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 19:27:58 2025

@author: Carola
"""

#ts4
#relación señal a ruido (???????????)
#snr, ventaneo
#parametrizar snr (antes era opcional ahora es obligatorio)
# estimadores, pengo nproceso estocástico que tiene parámetros, desconozco los valores concretos porque hay ruidoy etc, entonces lo estimo
#con los datos que tengo. pero nunca voy a saber el valor exacto dependen de no se qué estadística
#tengo n datos y quiero hacer lo mejr para conocer algun valor del sistema
# en el módulo de la transformada de fourier voy a ir a buscar el estimado de la amplitud a1
#vamos a ir a pescar la energía solo en w0/2pi, eso e UN estimador
# para energía!lo que buscamos al diseñar u estimador es  el valor esperado y la varianza sean alg proporcionales a cero (idealmente exactamente cero=)
#cuando hago converger algo a un valor que me interesa lo estoy CALIBRANDO. lo que no se puede es cambiarle la exactitud.

#estimador de frecuenccia: matriz de señales, devuelve matriz
#1er objetivo, ponerle a x el snr, superposicio´n

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

v=1
N=1000
fs=N
df=fs/N
v_med=0
w0= np.pi/2
SNR=10
amp=np.sqrt(2)

def senial(fs, N, amp, frec, fase=0, v_medio=0):
    xx= amp* np.sin(2*np.pi*frec*t+fase)
    return xx

tt = N / fs
t = np.arange(0, tt, 1/fs)
                                    
fr= np.random.uniform(-2, 2)
#na= np.random.normal(v_med, v)
na = np.random.normal(v_med, v, N)
w1= w0+ fr*(2*np.pi/N)
k = np.arange(N)
# w0 es fs/4    

senial_1=senial(t=t, amp=amp, frec=fs/4)
potenciaruido=amp**2/(2*10**(SNR/10))
ruido=np.random.normal(0, np.sqrt(potenciaruido), N)
varianzaruido=np.var(ruido)

x_1= senial_1 + ruido
xfft=fft(senial_1)
plt.figure(figsize=(10,5))
plt.plot()
# plt.plot(k, senial, 'o-', label='x(k)')
# plt.title("Señal $x(k) = a_0 \cdot \sin(\Omega_1 k) + n_a(k)$")
# plt.xlabel("k (muestras)")
# plt.ylabel("Amplitud")
# plt.grid(True)
# plt.legend()
# plt.show()



