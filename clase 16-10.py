# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:39:36 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# plantilla de diseño
wp= 1 #freciencia de corte/paso en rad/seg
ws= 5 #frecuencia de stop/detenida (rad/seg)

alpha_p = 3 #atenuacion maxima a la wp, alfa_max, pérdidas en banda de paso en (dB)
alpha_s = 40 #atenuacion minima a la ws, alfa_min, mínima atenuación requerida (dB)

#aprox módulo
# f_aprox= 'butter'
f_aprox= 'cheby1'
# f_aprox= 'cheby2'
# f_aprox= 'cauer' 

#aprox fase
# f_aprox= 'bessel'

#si le pido bessel aproxima la fase, las otras módulo
wc = 2 * np.pi * 1000  # frecuencia de corte (rad/s), ej. 1 kHz

# ... Diseño del filtro Butterworth analógico ...
b, a = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=True, ftype= f_aprox, output= 'ba')


#### Respuesta en frecuencia
w, h = signal.freqs(b, a, worN=np.logspace(-1, 2, 1000))
# w, h = signal.freqs(b, a) #calcula la respuesta en frecuencia del filtro
#H es un numero complejo
 # 10 Hz a 1 MHz aprox.  # 10 Hz a 1 MHz aprox.

#logspace es un espacio logarítmicamente espaciado
#en este caso hace uno entre 10 a la 1, 10 a la 6 y le mete 1000 espacios en el medio

#### Cálculo de fase y retardo de grupo

phase = np.unwrap(np.angle(h)) #si es una discontinuidad evitable la corrige, desenvuelve la periodicidad de la fase

# Retardo de grupo = -dφ/dω
gd = -np.diff(phase) / np.diff(w)

#### Polos y ceros
z, p, k = signal.tf2zpk(b, a) #pasamos a polos y ceros
#p es la localizacion de los ceros? polos?

###Gráficas

plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)), label=f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()
# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(phase), label=f_aprox)
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()
# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd, label=f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')
plt.legend()
# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#%%


#quiero pasar a celdas como las que yo ya conozco, no me sirve un orden grande, lo quiero factorizar
sos = signal.tf2sos(b, a)
