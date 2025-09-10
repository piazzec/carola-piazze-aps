# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 21:44:03 2025

@author: Carola
"""

#verificaciones de parseval

#debería asignarle raíz de dos a la amp y verificar que la varianza vale 1
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


N = 1000  
fs = N   
df = fs/N 
ts = 1/fs 

def sen(ff, nn, amp=1, dc=0, ph=0, fs=2):
    Nn = np.arange(nn)
    t = Nn/fs
    x = dc + amp * np.sin(2 * np.pi * ff * t + ph)
    return t, x


t, x = sen(ff=(N/4)*df,amp=np.sqrt(2) , nn=N, fs=fs)


var= np.var(x) 
media=np.mean(x)
desvio=np.std(x)
    
# t1, x1 = sen(ff=(N/4)*df, nn=N, fs=fs)
# t2, x2 = sen(ff=((N/4)+1)*df, nn=N, fs=fs)
# t2, x3 = sen(ff=((N/4)+0.5)*df, nn=N, fs=fs)

#transformada
xx1=fft(x)
xx1abs=np.abs(xx1)
xx1ang=np.angle(xx1)
xx1_cuadrado=xx1abs**2

# xx2=fft(x2)
# xx2abs=np.abs(xx2)
# xx2ang=np.angle(xx2)

# xx3=fft(x3)
# xx3abs=np.abs(xx3)
# xx3ang=np.angle(xx3)

Ft=np.arange(N)*df

plt.figure(1)
plt.plot(Ft,np.log10(xx1abs)*20,'x',label='x1 abs en db')
# plt.plot(Ft,np.log10(xx2abs)*20,'*',label='x2 abs en db')
# plt.plot(Ft,np.log10(xx3abs)*20,'*',label='x3 abs en db')
plt.figure(2)
plt.plot(xx1abs, 'x',label='x1 abs')
# plt.plot(xx2abs, '*',label='x2 abs')
plt.figure(3)
plt.plot(Ft,np.log10(xx1_cuadrado),'x', label='modulo cuadrado')

plt.title('FFT')
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Amplitud [dB]')

plt.legend()

plt.grid()
plt.tight_layout() 
plt.show()
