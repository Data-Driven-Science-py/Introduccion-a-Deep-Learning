import numpy as np
from numpy.typing import NDArray

def funcion_de_activacion(x: NDArray) -> NDArray:
    return np.max(x,0) ## max(x, 0) also named ReLU (non-linear)

def perceptron(x: NDArray) -> NDArray:
    w: NDArray = np.random.randn(x.shape[-1]) ## pesos predeterminados
    b: NDArray = np.random.randn(1) ## constante predeterminada
    o_t = np.dot(w, x) + b ## combinacion lineal
    y = funcion_de_activacion(o_t) ##evaluacion en la funcion de activacion
    return y ## retornar el valor

if __name__ == '__main__':
    # Ejemplo 1: Vector de entrada con valores positivos.
    x1 = np.array([1.5, 2.0, 3.0])
    print(f"Output for x1: {perceptron(x1)}")

    # Ejemplo 2: Vector de entrada con valores negativos.
    x2 = np.array([-1.0, -2.5, -3.0])
    print(f"Output for x2: {perceptron(x2)}")

    # Ejemplo 2: Vector de entrada con valores mezclados.
    x3 = np.array([0.5, -1.5, 2.0])
    print(f"Output for x3: {perceptron(x3)}")

    # Ejemplo 2: Vector de entrada con valore anulados.
    x4 = np.array([0.0, 0.0, 0.0])
    print(f"Output for x4: {perceptron(x4)}")
