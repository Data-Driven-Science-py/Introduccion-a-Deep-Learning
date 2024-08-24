## Retropopagacion - Backpropagation
La retropopagacion (o backpropagation) es un algoritmo basado en descenso de gradiente que permite la computacion de los gradientes de manera optimizada. Saca provecho de las propiedades de la regla de la cadena, es asi que dado el ejemplo del capitulo anterior, podemos determinar sus gradiente de la siguiente manera:

| ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Colored_neural_network_es.svg/800px-Colored_neural_network_es.svg.png) |

$x$: Entrada del modelo.

$a_1$: Salida de la primer capa.

$a_2$: Salida de la segunda capa.

$a_3$: Salida de la tercera capa.

Matrices $r \times c$ con los $c$ pesos de los $r$ perceptrones.

$w_1$: Peso de la primer capa.

$w_2$: Peso de la segunda capa.

$w_3$: Peso de la tercera capa.

Vectores con los $r$ biases/constantes de cada percpetron.

$b_1$: Peso de la primer capa.

$b_2$: Peso de la segunda capa.

$b_3$: Peso de la tercera capa.

## Propagacion hacia delante

$z_1 = x_1 w_1 + b_1$

$a_1 = \sigma(z_1)$

$z_2 = x_1 w_2 + b_2$

$a_2 = \sigma(z_2)$

$z_3 = x_1 w_3 + b_3$

$a_3 = \sigma(z_3)$


Para poder utilizar descenso de gradiente en estos parametros, necesitamos definir una funcion costo adecuada:

$J(w_1, w_2, w_3, b_1, b_2, b_3) = \frac{1}{n} \sum_{i=1}^{n} (f(x_i) - y_i)^2$

Tal que tengamos que encontrar la derivada parcial para cada una de las matrices para poder actualizar los pesos de forma optimizada.

$w_1 = w_1 - \gamma \frac{\partial J}{\partial w_1}$

$w_2 = w_2 - \gamma \frac{\partial J}{\partial w_2}$

$w_3 = w_3 - \gamma \frac{\partial J}{\partial w_3}$

## Retropopagacion
Es asi que podemos aprovecharnos de las propiedades de la derivada para no evaluarla muchas veces de manera innecesaria, entendamos el flujo de computaciones para poder implementar de manera optimizada este algoritmo:
1. Saquemos la derivada con respecto a la salida de la ultima capa:

$\frac{\partial J}{\partial a_3} = (a_3 - y)$

$\frac{\partial J}{\partial z_3} = (a_3 - y) \frac{\partial \sigma (z_3)}{z_3}$

$\frac{\partial J}{\partial w_3} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial w_3}$

$\frac{\partial J}{\partial a_2} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2}$

$\frac{\partial J}{\partial z_2} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2} \frac{\partial a_2}{\partial z_2}$

$\frac{\partial J}{\partial w_2} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2} \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial w_2}$

$\frac{\partial J}{\partial a_1} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2} \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1}$

$\frac{\partial J}{\partial z_1} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2} \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1} \frac{\partial a_1}{\partial z_1}$

$\frac{\partial J}{\partial w_1} = (a_3 - y) \frac{\partial \sigma (z_3)}{\partial z_3} \frac{\partial z_3}{\partial a_2} \frac{\partial a_2}{\partial z_2}\frac{\partial z_2}{\partial a_1} \frac{\partial a_1}{\partial z_1} \frac{\partial z_1}{\partial w_1}$

Notamos al instante que podemos utilizar las derivadas evaluadas en capas sucesivas para optimizar el calculo de capas anteriores. Es asi que en este caso, propagamos la red neuronal de manera inversa para poder ahorrar recursos y complejidad computacional mediante la reutilizacion de las derivadas parciales.

