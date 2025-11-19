# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:28:47 2025

@author: Carola
"""

"""
ts8

Método de spline cúbicos
con todos los algoritmos que vamos a ver la idea es disminuir el costo computacional
estimadores: 
    filtro de mediana, operador no lineal que opera en una ventana de 200ms, para cada muestra de ese largo calcula la mediana
    también hay filtro de media movil
    
    la mediana en contraposicion al valor medio es robusta
    
    a cada muestra le calculo la mediana en un lapso de 200ms, van a estar solapados
    es de200ms porque la idea es que con el soporte del filtro me pueda sacar de encima el ecg, para poder estimar la baja frecuencia
    el ancho de un complejo qrs normal está entre 70 y 150ms, mi soporte es bastante más ancho (200), elimina el complejo qrs porque lo toma como atípico (?)
    
    el de 200 filtra las transiciones abruptas/altas. todo lo que esté de 200 para abajo. el de 600 se lleva puestas las ondas p ? las otras del ecg digamos
    
    mi método de interpolacion (funcion interpolante) spline cúbico es suave, no tiene transiciones abruptas y resiste tres derivadas
    para interpolar necesitomis valores de referencia, después la funcion interpoladora  me va  a dar los puntos intermedios
    la onda p es la manifestacion de la compresion de las auriculas
    en el medio entre esto y el pico de compresión hay una pausa eléctrica. durante esa pausa eléctrica (fisiológicamente hay silencio) me dedico a tratar
    de estimar el ruido.
    a partir de los complejos qrs quiero tratar d detectar la onda p, y después el segmento pq (no vamos a hacer esto)
    
    a mi spline le paso el tiempo y la y 
    si me quiero sacar de encima una señal de 50 hz, q tiene una anchura de 20ms por el ancho de mi pq
    quiero que la respuesta en modulo de mi filtro tenga un cero en 50hz. 
    al promediar le voy poniendo ceros ?
    
    lo puedo pensar como un filtro lineal pero de tiempo variante, se va adecuando a lo que tenga que estimar. 
    antes de usar la funcion de interpolación me saco de encima el ruido
    
    # t_nuevo= np.linspace(t[0],t[-1],len(t)*10)
    # ECG_interp=cs(t_nuevo)

    # plt.figure()

    # plt.plot(t,ECG,'-',label='og')
    # #plt.plot(t_nuevo,ECG_interp,'-',label='spline interpolado')
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio

#----------------------------------------------------------
# Configuración general
#----------------------------------------------------------
fs = 1000  # [Hz]

###################################
# Lectura de señal ECG
###################################

# Mostrar contenido del archivo
sio.whosmat('ECG_TP4.mat')

# Cargar el archivo .mat
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
cant_muestras = N

###################################
# Tramo con ruido
###################################

x = ecg_one_lead[100000:120500]  # segmento con ruido

ecg_filt_mediana200 = sig.medfilt(x, 201)
ecg_filt_mediana600 = sig.medfilt(x, 601)

ecg_limpio = x - ecg_filt_mediana600

# --- Gráficos ---
plt.figure()
plt.plot(x, label='ECG con ruido')
plt.plot(ecg_limpio, label='ECG limpio (mediana)')
plt.plot(ecg_filt_mediana600, label='Filtro mediana (601)')
plt.legend()
plt.title('ECG con ruido')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [a.u.]')
plt.grid(True, ls=':')

###################################
# Tramo sin ruido
###################################

x = ecg_one_lead[4000:5500]  # segmento sin ruido

ecg_filt_mediana200 = sig.medfilt(x, 201)
ecg_filt_mediana600 = sig.medfilt(x, 601)

ecg_limpio = x - ecg_filt_mediana600

# --- Gráficos ---
plt.figure()
plt.plot(x, label='ECG sin ruido')
plt.plot(ecg_limpio, label='ECG limpio (mediana)')
plt.plot(ecg_filt_mediana600, label='Filtro mediana (601)')
plt.legend()
plt.title('ECG sin ruido')
plt.xlabel('Muestras')
plt.ylabel('Amplitud [a.u.]')
plt.grid(True, ls=':')
plt.show()



#plt.plot(ecg_filt_cauer[:50000], label= 'cauer')
#plt.plot(ecg_filt_cbeb1[:50000], label= 'cheb1')
#plt.plot(ecg_filt_cheb2[:50000], label= 'cheb2')


# plt. legend()
# ###################################
# #Regiones de interés sin ruido #
# ###################################
 
# regs_interes = (
#         [4000, 5500], # muestras
#         [10e3, 11e3], # muestras
#         )
 
# for ii in regs_interes:
   
#     # intervalo limitado de 0 a cant_muestras
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth')
#     #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
#     plt.title('ECG sin ruido desde' + str(ii[0]) + ' to ' + str(ii[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
   
#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())
           
#     plt.show()
 
# ###################################
# # Regiones de interés con ruido #
# ###################################
 
# regs_interes = (
#         np.array([5, 5.2]) *60*fs, # minutos a muestras
#         np.array([12, 12.4]) *60*fs, # minutos a muestras
#         np.array([15, 15.2]) *60*fs, # minutos a muestras
#         )
 
# for ii in regs_interes:
   
#     # intervalo limitado de 0 a cant_muestras
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     plt.plot(zoom_region, ecg_filt_butt[zoom_region], label='Butterworth')
#     #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
#     plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
   
#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())
           
#     plt.show()
