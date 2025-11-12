# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:35:02 2025

@author: Sofía
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio

## 1. Mediana
# b^ La idea es que el soporte del filtro me saque el ECG para estimar las bajas frecuencias - linea de base
# la med(s) de 200 se saca de encima el complejo QRS y el de 600 las ondas p y f

##%

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].squeeze()

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

## 2. Splines Cubicos
# la funcion interpolante es suave, C3. 
# Puntos de referencia, segmentos isoelectricos

#si sumo muestras tengo un pasabajo ??
