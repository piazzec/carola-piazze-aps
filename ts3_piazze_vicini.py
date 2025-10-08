# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:34:03 2025

@author: Carola
"""
#ts2 efecto del desparramo espectral

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


N = 1000       #cant muestras
fs = N         #frec de muestreo
df = fs/N      #resolución espectral
ts = 1/fs      #tiempo entre muestras

 
def sen(frecuencia, nn, amp=1, dc=0, fase=0, fs=2):
    Nn = np.arange(nn) #nn cantidad de muestras
    t = Nn/fs
    x = dc + amp * np.sin(2 * np.pi * frecuencia * t + fase)
    return t, x


amp = np.sqrt(2)   #normalización

t1, x1 = sen(frecuencia=(N/4)*df, nn=N, fs=fs, amp=amp)
t2, x2 = sen(frecuencia=((N/4)+0.25)*df, nn=N, fs=fs, amp=amp)
t3, x3 = sen(frecuencia=((N/4)+0.5)*df, nn=N, fs=fs, amp=amp)

#t fourier
xx1 = fft(x1); xx1abs = np.abs(xx1)
xx2 = fft(x2); xx2abs = np.abs(xx2)
xx3 = fft(x3); xx3abs = np.abs(xx3)



Ft = np.arange(N)*df #eje de frecuencias, va de 0 a fs-df


plt.figure(1)
plt.plot(Ft, 10*np.log10(xx1abs**2), 'x', label='k0=N/4')
plt.plot(Ft, 10*np.log10(xx2abs**2), '*', label='k0=N/4+0.25')
plt.plot(Ft, 10*np.log10(xx3abs**2), 'o', label='k0=N/4+0.5')
plt.title('Desparramo espectral')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()

#varianzas
print("Varianzas:")
print("k0=N/4:", np.var(x1))
print("k0=N/4+0.25:", np.var(x2))
print("k0=N/4+0.5:", np.var(x3))
print()


##B
#energía en tiempo
p1 = np.sum(np.abs(x1)**2)
p2 = np.sum(np.abs(x2)**2)
p3 = np.sum(np.abs(x3)**2)

#energía en frecuencia
P1 = (1/N) * np.sum(np.abs(xx1)**2)
P2 = (1/N) * np.sum(np.abs(xx2)**2)
P3 = (1/N) * np.sum(np.abs(xx3)**2)


# print("Señal 1:")
# print("Energía tiempo =", p1)
# print("Energía frecuencia =", P1)


# print("\nSeñal 2:")
# print("Energía tiempo =", p2)
# print("Energía frecuencia =", P2)


# print("\nSeñal 3:")
# print("Energía tiempo =", p3)
# print("Energía frecuencia =", P3)


tol = 1e-10

# Señal 1
p1 = np.sum(np.abs(x1)**2)
P1 = (1/N) * np.sum(np.abs(xx1)**2)
if np.abs(p1 - P1) < tol:
    print("k0=N/4: Se cumple Parseval")
else:
    print("k0=N/4: No se cumple Parseval")

# Señal 2
p2 = np.sum(np.abs(x2)**2)
P2 = (1/N) * np.sum(np.abs(xx2)**2)
if np.abs(p2 - P2) < tol:
    print("k0=N/4+0.25: Se cumple Parseval")
else:
    print("k0=N/4+0.25: No se cumple Parseval")

# Señal 3
p3 = np.sum(np.abs(x3)**2)
P3 = (1/N) * np.sum(np.abs(xx3)**2)
if np.abs(p3 - P3) < tol:
    print("k0=N/4+0.5: Se cumple Parseval")
else:
    print("k0=N/4+0.5: No se cumple Parseval")

print()




##C
padding = np.zeros(9*N)  #9N ceros

#pego las funciones ala lista de ceros
x1_pad = np.concatenate([x1, padding])
x2_pad = np.concatenate([x2, padding])
x3_pad = np.concatenate([x3, padding])

#a la nueva función le hago la transformada
xx1_pad = fft(x1_pad)
xx2_pad = fft(x2_pad)
xx3_pad = fft(x3_pad)

#
XX1_pad = np.abs(xx1_pad)**2
XX2_pad = np.abs(xx2_pad)**2
XX3_pad = np.abs(xx3_pad)**2

# Nuevo eje de frecuencias
M = len(x1_pad)      #10N
df_pad = fs/M
Ft_pad = np.arange(M)*df_pad


# plt.figure(2)
# plt.plot(Ft_pad, 10*np.log10(XX1_pad), 'x', label='k0=N/4')
# plt.plot(Ft_pad, 10*np.log10(XX2_pad), '*', label='k0=N/4+0.25')
# plt.plot(Ft_pad, 10*np.log10(XX3_pad), 'o', label='k0=N/4+0.5')
# plt.title('Desparramo espectral con zero padding')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Potencia [dB]')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
plt.figure(2)
plt.plot(Ft_pad, 10*np.log10(XX1_pad), label='k0=N/4')
plt.plot(Ft_pad, 10*np.log10(XX2_pad), label='k0=N/4+0.25')
plt.plot(Ft_pad, 10*np.log10(XX3_pad),  label='k0=N/4+0.5')
plt.title('Desparramo espectral con zero padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

