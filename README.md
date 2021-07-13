---
### Universidad de Costa Rica
#### IE0405 - Modelos Probabilísticos de Señales y Sistemas
---
* Estudiante: *Adrián Avilés Flores*
* Carné: *B80835*
* Grupo: *2*
---
# `P4` - *Modulación digital IQ*
---
## Modulación 16-QAM

### Imagen transmitida e imagen recuperada
![](Modulacion.jpg)

### Onda transmitida y onda transmitida con ruido
![](Ondas.jpg)

#### En esta sección del proyecto se procedió a simular el procedimiento de transmisión de imágenes mediante la modulación 16-QAM, para ello fue necesario realizar una nueva función moduladora y demoduladora que trabajaran con 4 bits por símbolo. Para la prueba anteriormente realizada se utilizó una relación señal-a-ruido del canal (SNR) de -5 dB, por lo que la onda transmitida tuvo una afectación considerable y desembocó en una imagen recuperada con errores, pero se puede notar la gran similitud entre la imagen inicial y la final.
---
### Estacionaridad y ergodicidad

#### Se puede determinar que la señal modulada que transmite la imagen consiste en un proceso aleatorio estacionario en sentido amplio puesto que la modulación es de la forma: s(t) = A_1 cos(2*pi f_c t) + A_2 sin(2*pi f_c t) y se sabe que los promedios temporales de las componentes coseno y seno serán de 0.

#### Ahora, como se conoce que este proceso aleatorio es estacionario en sentido amplio, también se puede decir que es ergódico puesto que los promedios temporales van a ser iguales que los promedios estadísticos.
---
## Densidad espectral de potencia

### Densidad obtenida

![](Densidad.jpg)

#### En esta sección se debe utilizar la transformada de Fourier y la fórmula provista en el enunciado para determinar la densidad espectral de potencia, para el caso de la modulación 16-QAM se puede determinar que la frecuencia central se da a los 5 kHz (como era esperado) y también se presentan otras frecuencias secundarias que son propias de la modulación 16-QAM.
