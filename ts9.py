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
#%%
########################
#ITEM 3- FILTO ADAPTADO 
########################

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


qrs_pattern1= mat_struct['qrs_pattern1']
pattern2= qrs_pattern1.flatten() - np.mean(qrs_pattern1)#para poder correlar


ecg_detection=signal.lfilter(b=pattern2,a=1, x=ecg_one_lead)
ecg_detection_abs=np.abs(ecg_detection)[57:]
ecg_detection_norm = ecg_detection_abs/np.std(ecg_detection_abs)

qrs, _ = signal.find_peaks(ecg_detection_norm, height=1, distance=300)
ecg_one_lead_abs=np.abs(ecg_one_lead)
ecg_one_lead_norm=ecg_one_lead_abs/np.std(ecg_one_lead_abs)

plt.figure()
plt.plot(ecg_detection_norm)
plt.plot(qrs, ecg_detection_norm[qrs], 'r*',label='qrs')
plt.plot(ecg_one_lead_norm)
plt.legend()
plt.title('detección de latidos')
plt.show()

#%%
#copiar prominencia
qrs, promin=signal.find_peaks(ecg_detection_norm)
######################
qrs_det=mat_struct['qrs_detections']

def matriz_confusion_qrs(qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    qrs = np.array(qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0 # True Positives
    FP = 0 # False Positives
    FN = 0 # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    FN = np.sum(~qrs_det_emparejados)
    
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0] # TN generalmente no aplica en detección de eventos
    ])
    
    return matriz, TP, FP, FN

# Ejemplo de uso

matriz, tp, fp, fn = matriz_confusion_qrs(qrs, qrs_det)

print("Matriz de Confusión:")
print(f" Predicho")
print(f" Sí No")
print(f"Real Sí: [{tp:2d} {fn:2d}]")
print(f"Real No: [{fp:2d} - ]")
print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")

# Calcular métricas de performance
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nMétricas:")
print(f"Precisión: {precision:.3f}")
print(f"Sensibilidad: {recall:.3f}")
print(f"F1-score: {f1_score:.3f}")

#esto grafica todos los latidos (la r) juntos. me muestra los picos que encontré
qrs_mat = [ecg_one_lead[ii-60:ii+60] for ii in qrs] 
qrs_mat = np.array([ecg_one_lead[ii-60:ii+60] for ii in qrs])
qrs_mat = qrs_mat - np.mean(qrs_mat)
np.mean(qrs_mat)
np.mean(qrs_mat,axis=1).shape
np.mean(qrs_mat,axis=0).shape
qrs_mat = qrs_mat - np.mean(qrs_mat, axis = 1).reshape((-1,1)) 
np.mean(qrs_mat,axis=0).shape
np.mean(qrs_mat,axis=1).reshape((-1,1)).shape
qrs_mat = qrs_mat - np.mean(qrs_mat, axis = 1).reshape((-1,1))
plt.plot(qrs_mat.transpose())

