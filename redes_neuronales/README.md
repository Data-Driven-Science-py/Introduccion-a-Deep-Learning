# Redes neuronales

| ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Colored_neural_network_es.svg/800px-Colored_neural_network_es.svg.png) |
|:--:|
| *Figura 1. Ejemplo de una red neuronal.* |

La red neuronal es un algoritmo compuesto por la iteracion secuencial de evaluaciones de perceptrones. Podria interpretarse como una representacion artificial del traspaso de informacion neuronal.

## Conceptos

**Capas**: Son las configuraciones verticales de la red neuronal, en el caso del ejemplo tenemos tres capas.
**Perceptrones**: Explicado en el capitulo anterior, en el caso del ejemplo, en la primera capa tenemos 3 perceptrones, en la segunda 4, y en la ultima 2.
**Entrada**: Es la capa de entrada.
**Capas escondidas**: Son las capas que se encuentran en la capa de entrada y la final.
**Salida**: Es la capa de salida.

## Funcionamiento
Dado un vector de entrada $x$, digamos que este vector es de tres dimensiones: $x = (x_1, x_2, x_3)$, este vector es pasado como entrada a cada uno de los perceptrones de la capa de entrada, los resultados de cada uno de los perceptrones se concatenan en un vector escondido que va a ser la entrada de la siguiente capa de perceptrones.
Una red neuronal tiene capas que dada la evaluacion secuencial, su computacion depende de que capas anteriores terminen su trabajo. Por el otro lado, perceptrones de la misma capa pueden ser evaluados de manera independiente con respecto a los demas.
Para poder interpretar, dese el caso de la figura 1, primero tenemos dos perceptrones, ya sabemos como funcionan, dado el vector de entrada $x$ que tenemos, terminara evaluando esta entrada de la siguiente manera:
$f_1(x) = w_{11} x_1 + w_{12} x_2 + w_{13} x_3 + b = a_{11}$
$f_2(x) = w_{21} x_1 + w_{22} x_2 + w_{23} x_3 + b = a_{12}$
$f_3(x) = w_{31} x_1 + w_{32} x_2 + w_{33} x_3 + b = a_{13}$

El primer valor del indice representa el numero del perceptron dentro de la capa, y el segundo es el indicador del peso dentro del perceptron (En el caso de $w$). El indice de $a$ representa que es la salida de la primera capa (1) y luego el indicativo de su proveniencia con respecto a los perceptrones.

Los resultados de los dos perceptrones se juntan en un nuevo vector de entrada para la siguiente capa:
$a_1 = (a_{11}, a_{12})$

Y este vector termina siendo pasado a los perceptrones de la siguiente capa para repetir este proceso de manera iterada.

