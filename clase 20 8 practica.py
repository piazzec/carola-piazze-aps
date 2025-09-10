# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:33:32 2025

@author: Carola
"""

import numpy as np
import matplotlib.pyplot as plt

#Clase 20/8/2025
#suma de N términos -> 
N=100
fs=1
vc=1
ff=N

x=np.zeros(N)
x[0]=1 #defino delta
x1=np.sin(vc=0, fs=0, ph=0, nn=N, ff=ff) #defino funcion senoidal con frecuencia que debería coincidir (esta definicion está mal)
n = np.arange(N)/ff
#debería hacer una funcion transformada dft y llamarla
X=np.zeros(N, dtype=np.complex128) 
for k in range(N):
    for n in range(N):
        X+=x1*np.exp(-1j*k*((2*np.pi)/N)*n) #esta es mi tranformada, la tengo que aplicar
        
#para ver si esto anda puedo probar pimplementando con una delta que ya sé que es 1
#tengo que hacer un gráfico para ver que está pasando, para esto CREO que tengo que calcular el módulo 
#los k giran más lento
#como calcular el módulo de cada elemento complejo
#cuando ponemos una senoidal esperamos que nos de dos deltas, que tienen que proyectar con la parte imaginaria ???
#x[k] debería proyectar algo completamente imaginario
#si hago un coseno tengo que lograr que la proyección sea imaginaria
plt.figure
plt.plot(n, x1, color="black")
plt.grid(True)
plt.show()


#sigue clase explica algo de fase
#la fase es el arctg de la prte imaginaria de x sobre la parte real de x
#la función np.angle es una función de 4 cuadrantes