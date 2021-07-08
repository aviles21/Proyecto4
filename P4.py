import numpy as np
import matplotlib.pyplot as plt
import time

# Modulación 16-QAM

from PIL import Image
import numpy as np

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

def modulador_16_QAM(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits
    
    # Datos que se modularan con coseno
    DatosA = []
    
    # Datos que se modularan con seno
    DatosB = []
    
    # Suma los primeros 2 bits de cada simbolo de 4 bits y los guarda en DatosA
    # También suma los últimos 2 bits de cada símbolo y los guarda en DatosB
    for k in range(int(N/4)):
        DatosA.append(2*bits[k*4] + bits[1+(k*4)])
        DatosB.append(2*bits[2+(k*4)] + bits[3+(k*4)])

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora_sin = np.sin(2*np.pi*fc*t_periodo)
    portadora_cos = np.cos(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, int(N/4)*Tc, int(N/4)*mpp) 
    senal_Tx_sin = np.zeros(t_simulacion.shape)
    senal_Tx_cos = np.zeros(t_simulacion.shape)
    senal_Tx = np.zeros(t_simulacion.shape)
 
    # 4. Asignar las formas de onda según los bits (16-QAM)
    for i, bit in enumerate(DatosA):
        if bit == 0:
            senal_Tx_cos[i*mpp : (i+1)*mpp] = portadora_cos * -3
            
        elif bit == 1:
            senal_Tx_cos[i*mpp : (i+1)*mpp] = portadora_cos * -1
        
        elif bit == 3:
            senal_Tx_cos[i*mpp : (i+1)*mpp] = portadora_cos * 1
        
        else:
            senal_Tx_cos[i*mpp : (i+1)*mpp] = portadora_cos * 3
    
    for i, bit in enumerate(DatosB):
        if bit == 0:
            senal_Tx_sin[i*mpp : (i+1)*mpp] = portadora_sin * 3
            
        elif bit == 1:
            senal_Tx_sin[i*mpp : (i+1)*mpp] = portadora_sin * 1
        
        elif bit == 3:
            senal_Tx_sin[i*mpp : (i+1)*mpp] = portadora_sin * -1
        
        else:
            senal_Tx_sin[i*mpp : (i+1)*mpp] = portadora_sin * -3
    
    # Se combinan las señales
    senal_Tx = senal_Tx_cos + senal_Tx_sin
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora_cos, portadora_sin

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

def demodulador_16_QAM(senal_Rx, portadora_cos, portadora_sin, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de símbolos en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N*4)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora_sin * portadora_sin)

    # Demodulación
    
    # Recupera los datos que se encontraban en DatosA
    for i in range(N):
        # Producto interno de dos funciones
        producto_cos = senal_Rx[i*mpp : (i+1)*mpp] * portadora_cos
        Ep_cos = np.sum(producto_cos)
        
        if Ep_cos < -20:
            bits_Rx[i*4] = 0
            bits_Rx[1+i*4] = 0
        
        elif 0 > Ep_cos > -20:
            bits_Rx[i*4] = 0
            bits_Rx[1+i*4] = 1
        
        elif 0 < Ep_cos < 20:
            bits_Rx[i*4] = 1
            bits_Rx[1+i*4] = 1
            
        elif Ep_cos > 20:
            bits_Rx[i*4] = 1
            bits_Rx[1+i*4] = 0
            
    # Recupera los datos que se encontraban en DatosB
    for i in range(N):
        # Producto interno de dos funciones
        producto_sin = senal_Rx[i*mpp : (i+1)*mpp] * portadora_sin
        Ep_sin = np.sum(producto_sin)
        
        if Ep_sin < -20:
            bits_Rx[2+i*4] = 1
            bits_Rx[3+i*4] = 0
        
        elif 0 > Ep_sin > -20:
            bits_Rx[2+i*4] = 1
            bits_Rx[3+i*4] = 1
        
        elif 0 < Ep_sin < 20:
            bits_Rx[2+i*4] = 0
            bits_Rx[3+i*4] = 1
            
        elif Ep_sin > 20:
            bits_Rx[2+i*4] = 0
            bits_Rx[3+i*4] = 0

    return bits_Rx.astype(int)

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora_cos, portadora_sin = modulador_16_QAM(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx = demodulador_16_QAM(senal_Rx, portadora_cos, portadora_sin, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# Visualizar el cambio entre las señales
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(14, 7))

# La señal modulada por BPSK
ax1.plot(senal_Tx[0:600], color='g', lw=2) 
ax1.set_ylabel('Señal transmitida')
ax1.set_title('Formas de onda')

# La señal modulada al dejar el canal
ax2.plot(senal_Rx[0:600], color='b', lw=2) 
ax2.set_ylabel('Señal recuperada con ruido')
ax2.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

# Densidad espectral de potencia

from scipy import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.title("Densidad espectral de potencia")
plt.show()
