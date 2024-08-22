# Perceptron
Unidad de computo a quien se le asigna una cantidad determinada de parametros para hacer una computacion determinada.

## Ejemplo:

| ![image](https://miro.medium.com/v2/resize:fit:1290/1*-JtN9TWuoZMz7z9QKbT85A.png) |
|:--:|
| *Figura 1. Representacion grafica de un perceptron. (Chartran, 2018)* |


$y = x_3*w_3 + x_2*w_2 + x_1*w_1$

# Generalizacion

| ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Perceptr%C3%B3n_5_unidades.svg/400px-Perceptr%C3%B3n_5_unidades.svg.png) |
|:--:|
| *Figura 2. Representacion grafica de un perceptron. (Wikipedia, 2015)* |

**Definicion 1**: Un perceptron de cantidad de pesos $n$ es una unidad de computo que genera una combinacion lineal de una serie de valores de entrada ($x_i$) con pesos predeterminados($w_i$):

$O^t = \sum_{i=1}^{n} x_i w_i + b$

seguido de la inclusion de alguna funcion lineal o no lineal llamada **funcion de activacion**.

$y = \sigma(O^t)$

| ![image](https://www.researchgate.net/publication/313238723/figure/fig1/AS:457343222718466@1486050529523/Figura-1-Comparacion-entre-una-neurona-biologica-y-una-neurona-artificial.png) |
|:--:|
| *Figura 3. Comparacion entre una neurona biologica y una neurona artificial. (Leon, 2017)* |

La creacion del perceptron esta fuertemente ligada a la representacion de un algoritmo discriminante que imite el funcionamiento de las neuronas humanas, es asi que muchos de los avances que hacen posible a la inteligencia artificial vienen de estudios inter-disciplinarios.


