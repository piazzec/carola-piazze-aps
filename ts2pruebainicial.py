# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:27:51 2025

@author: Carola
"""

''' tengo que hallar y[n] con cada una de mis señales x del otro tp, luego transformar para despejar h de la convolución (y=h*x)
una vez que encuentro h[n] puedo calcular la salida y[n] para cualquier otra entrada
como me dice que son causales, yn depende solo de xk, y la salida es igual a la entrada
Sistema LTI (Lineal e Invariante en el tiempo): definido por su respuesta al impulso 
h[n]. La salida para cualquier entrada causal viene por convolución:
'''
'''
Un sistema LTI (Lineal e Invariante en el Tiempo) puede modelarse mediante ecuaciones en diferencias con coeficientes constantes. La ecuación dada:
representa un sistema recursivo (IIR) donde la salida actual depende de entradas actuales, pasadas y salidas pasadas.

La respuesta al impulso 
h[n] de un sistema LTI es la salida cuando la entrada es un impulso unitario 
δ[n]. Para sistemas IIR, la respuesta al impulso tiene duración infinita, pero en la práctica se aproxima con una longitud finita.

La salida de un sistema LTI puede calcularse mediante la convolución entre la entrada y la respuesta al impulso:
Para señales discretas, se implementa con np.convolve.
    '''
import numpy as np
import matplotlib.pyplot as plt


fs = 60000      
Ts = 1 / fs        

#copio señales ts1
t_seno = np.linspace(0, 0.002, int(fs*0.002), endpoint=False)   # 2 ms
t_cuadrada = np.linspace(0, 0.005, int(fs*0.005), endpoint=False)  # 5 ms
t_pulso = np.linspace(0, 0.02, int(fs*0.02), endpoint=False)     # 20 ms

frec = 2000.0  # Hz
x1 = np.sin(2*np.pi*frec*t_seno)                      # seno 2 kHz
x2 = 2*np.sin(2*np.pi*frec*t_seno + np.pi/2)          # A=2, desfase pi/2
frec_mod = 1000.0
m = 0.8
x3 = (1 + m*np.sin(2*np.pi*frec_mod*t_seno)) * np.sin(2*np.pi*frec*t_seno)  # AM
A = np.max(np.abs(x1))
x1_clipped = np.clip(x1, -0.75*A, 0.75*A)
frec_cuadrada = 4000.0
x_cuadrada = np.sign(np.sin(2*np.pi*frec_cuadrada*t_cuadrada))



# guardo las señales en un diccionario para iterar
señales = {
    'x1_seno': (x1, t_seno),
    'x2_seno_A2_desfase': (x2, t_seno),
    'x3_AM': (x3, t_seno),
    'x1_clipped': (x1_clipped, t_seno),
    'x_cuadrada': (x_cuadrada, t_cuadrada),

}

#funciones auxiliares

def calc_potencia(x): #(para periódicas)
    return np.mean(x**2)

def calc_energia(x, Ts): #(para finitas)
    return np.sum(x**2) * Ts

#tener en cuenta que como es causal todo lo que pase antes (ejemplo y[n-1]) cuando n es 0 debe ser cero, por eso las condiciones del for
def LTI_1(x):
    N = len(x)         
    y = np.zeros(N) 
    #inicializo el vector de ceros como en clase

    # caso n=0
    y[0] = 0.03 * x[0]

    # caso  n=1
    if N > 1:  # para que no se rompa todo si la señal es muy corta
        y[1] = 0.03*x[1] + 0.05*x[0] + 1.5*y[0]

    # cuando n >= 2
    for n in range(2, N):
        y[n] = (0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2]
                 + 1.5*y[n-1] - 0.5*y[n-2])
    return y

# convolución a mano para después comparar con umpy
def conv_amano(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    
    for n in range(len(y)):
        for k in range(max(0, n - M + 1), min(n + 1, N)):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n - k]
    return y[:N]  # Recortar al tamaño original

#respuesta al impulso
def impulso(sistema, N_impulso = 100 ):
    delta = np.zeros(N_impulso)
    delta[0] = 1
    h = sistema(delta)
    return h

# ======================== ITEM 1

# recorro las señales definidas yvoy aplicando el lti
for nombre, (x, t) in señales.items():
    y = LTI_1(x)  # aplico
    #info de la señal
    duracion = t[-1] + Ts
    print(f"\nSeñal: {nombre}")
    print(f"  Frecuencia de muestreo: {fs} Hz")
    print(f"  Duración: {duracion:.6f} s")
    # gráfico comparando entrada y salida
    plt.figure(figsize=(10, 6))
    plt.plot(t*1000, x, '-.',color='mediumseagreen', label="Entrada " + nombre)
    plt.plot(t*1000, y,'b', label="Salida y[n]")
    plt.xlabel("Tiempo [ms]")
    plt.title("Sistema LTI aplicado a " + nombre)
    plt.legend()
    plt.grid(True)
    plt.show()

    # cálculos de energia y ppotencia
    if nombre == 'x_cuadrada' or nombre.startswith('x1_seno'): 
        #idea copada de startswith de chatgpt, no la tenía y me simplifica las cosas
        pot = calc_potencia(y)
        print(f"Potencia de salida para {nombre}: {pot:.4f}")
    else:
        ener = calc_energia(y, Ts)
        print(f"Energía de salida para {nombre}: {ener:.4f}")





#respuesta al impulso
h=impulso(LTI_1)
n = np.arange(len(h))
plt.figure(figsize=(10, 6))
plt.plot(n, h, 'bo', label='h[n]')   
plt.vlines(n, 0, h, colors='b')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso')
plt.grid(True)
plt.legend()
plt.show()


#comparación de convolución
# aplico a señal x1
y_amano = conv_amano(x1, h)
y_numpy = np.convolve(x1, h, mode='same')
y_lti = LTI_1(x1)

# Graficar comparación
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t_seno*1000, y_lti, 'b-')
plt.title('Salida con el método directo')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_seno*1000, y_amano, 'r-')
plt.title('Salida con la convolución a mano')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_seno*1000, y_numpy, 'g-')
plt.title('Salida con la convolución con numpy')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#revisar que hay un gráfico que tiene la leyenda en el medio


# ======================== ITEM 2


def sist_1(x):
    N = len(x)
    y = np.zeros(N)
    
    for n in range(N):
        if n < 10:
            y[n] = x[n] 
        else:
            y[n] = x[n] + 3 * x[n-10]
    
    return y


def sist_2(x):
    N = len(x)
    y = np.zeros(N)
    
    for n in range(N):
        if n < 10:
            y[n] = x[n] 
        else:
            y[n] = x[n] + 3 * y[n-10]
    
    return y   

# impulsos, uso x2 para variar aunque es práticamente igual
hh=impulso(sist_1)
hhh=impulso(sist_2)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
n = np.arange(len(hh))
plt.plot(n, hh, 'mo')  
plt.vlines(n, 0, hh, colors='magenta')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso hh')
plt.grid(True)
plt.legend()
plt.show()

plt.subplot(2, 1, 2)
n = np.arange(len(hhh))
plt.plot(n, hhh, 'go')  
plt.vlines(n, 0, hhh, colors='green')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Respuesta al impulso hhh')
plt.grid(True)
plt.legend()
plt.show()

plt.tight_layout()
plt.show()

#salidas
y1 = sist_1(x2)
y2 = sist_2(x2)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_seno*1000, y1, 'b-')
plt.title('Salida de x2 con ecuacion de diferencias 2')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_seno*1000, y2, 'g-')
plt.title('Salida de x2 con ecuacion de diferencias 3')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#FALTA ENERGÍA EN ALGUN LADO REVISAR
