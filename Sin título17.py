# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:35:02 2025

@author: carola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio
from scipy.interpolate import CubicSpline

## 1. Mediana
# b^ La idea es que el soporte del filtro me saque el ECG para estimar las bajas frecuencias - linea de base
# la med(s) de 200 se saca de encima el complejo QRS y el de 600 las ondas p y f

##%

sio.whosmat('ecg.mat')
mat_struct = sio.loadmat('./ecg.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
#qrst detenctions, calcular cuanto me tengo que mover desde la linea roja hasta la zona de deteccion

ECG=ecg_one_lead [100000: 120500] #Tramo con Ruido


med200=signal.medfilt(ECG, 201)
med600=signal.medfilt(ECG, 601) #b

ECG_limpio=ECG-med600

plt.plot(ECG, label='ECG')
plt.plot(ECG_limpio, label='ECG Limpio')
plt.plot(med600, label='b')
plt.legend()

ECG=ecg_one_lead [4000: 5500] #Tramo sin Ruido


med200=signal.medfilt(ECG, 201)
med600=signal.medfilt(ECG, 601) #b

ECG_limpio=ECG-med600

plt.figure()
plt.plot(ECG, label='ECG')
plt.plot(ECG_limpio, label='ECG Limpio')
plt.plot(med600, label='b')
plt.legend()
#%%

################
#SPLINE CÚBICO
################ 

## 2. Splines Cubicos
# la funcion interpolante es suave, C3. 
# Puntos de referencia, segmentos isoelectricos

#si sumo muestras tengo un pasabajo ??

#class CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)[source]
#el cubicspline no devuelve una variable devuelve objetos, son instancias ??que tiene distintos operadores? que hacen cosas. me devuelve une 
#referencia al objeto


maximos= mat_struct['qrs_detections']
vector_max=maximos-80 #ahora tengo todos los valores e el medio de ese valle entre p y q
#implementación
sx=vector_max.flatten()
sy=ecg_one_lead[sx]
N=len(ecg_one_lead)
cs=CubicSpline(sx,sy)
n=0
b=cs(np.arange(N))

plt.figure()
plt.scatter(sx,sy,color='red',marker='x')
plt.plot(b,label='b, cubic spline')
plt.plot(ecg_one_lead, color='lightgreen',label='ecg')
plt.show()

########################
#ITEM 3- FILTO ADAPTADO 
########################
#%%
qrs_pattern1= mat_struct['qrs_pattern1']
pattern2= qrs_pattern1.flatten() - np.mean(qrs_pattern1)

ecg_detection=signal.lfilter(b=pattern2,a=1, x=ecg_one_lead)
ecg_detection_abs=np.abs(ecg_detection)

"""
n Filtro adaptado es un sistema lineal invariante cuya función principal es detectar la presencia de una señal conocida, 
o referencia, dentro de una señal recibida. La señal a la salida del filtro será la correlación de la señal referencia con
 la señal desconocida. Esto es equivalente a realizar la convolución de la señal desconocida con una versión invertida de l
 a referencia (que además tiene un desplazamiento t0). Por propiedad de la convolución, también es equivalente a realizar
 la convolución de la señal de referencia con una versión invertida x(-t) de la señal desconocida.

Cuanto más corto sea el ?? más cercana a cero va a ser la salida
un filtro pasabajo puede ser un detector de envolvente

la mejor forma de estimaruna señal en ruido es con un filtro adaptado
para conocer el patron creo un protocolo 
"""

#la derivada en las altas frecuencias va a ser más importante (más grande) que en las bajas frecuencias, por cómo se comortan
#las derivadas en las transformadas

#protopromediador

# salida del filtro adaptado
#ecg_detection = sig.lfilter(b=pattern[::-1], a=1, x=ecg_one_lead)

# 1. valor absoluto (y compensar retardo)
ecg_detection_abs = np.abs(ecg_detection)[57:]

# 2. normalizar
ecg_detection_abs = ecg_detection_abs / np.std(ecg_detection_abs)

# 3. filtro pasabajo tipo integrador
ecg_detection_abs_lp = signal.lfilter(b=np.ones(111), a=1, x=ecg_detection_abs)

# 4. detección de picos (usando distancia mínima)
qrs, _ = signal.find_peaks(ecg_detection_abs_lp, height=3, distance=200)

# visualizar
plt.figure(figsize=(12,4))
plt.plot(ecg_detection_abs_lp)
plt.plot(qrs, ecg_detection_abs_lp[qrs], 'r*', label='QRS detectados')
plt.legend()

plt.show()


