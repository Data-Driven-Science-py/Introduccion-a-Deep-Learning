import numpy as np
from numpy.typing import NDArray

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def loss(input, target) -> NDArray:
    return np.mean((input - target)**2)

class RedNeuronal:
    def __init__(self, input_size: int):
        self.w_1 = np.random.randn(input_size, 3) # Pesos de la primera capa
        self.w_2 = np.random.randn(3, 4) # Pesos de la segunda capa
        self.w_3 = np.random.randn(4, 2) # Pesos de la tercera capa

        self.b_1 = np.random.randn(1, 3) # Bias de la primera capa
        self.b_2 = np.random.randn(1, 4) # Bias de la segunda capa
        self.b_3 = np.random.randn(1, 2) # Bias de la tercera capa

    def propagacion_hacia_delante(self, x: NDArray) -> NDArray:
        self.z_1 = np.dot(x, self.w_1) + self.b_1
        self.a_1 = sigmoid(self.z_1)
        self.z_2 = np.dot(self.a_1, self.w_2) + self.b_2
        self.a_2 = sigmoid(self.z_2)
        self.z_3 = np.dot(self.a_2, self.w_3) + self.b_3
        self.a_3 = sigmoid(self.z_3)
        return self.a_3

    def retropropagacion(self, x: NDArray, y: NDArray, learning_rate: float):
        # Calcular el error general y la gradiente con respecto al output
        error_salida = y - self.a_3
        d_salida = error_salida * d_sigmoid(self.z_3)

        # Calcular el error de la
        error_escondida_2 = d_salida.dot(self.w_3.T)
        d_escondido_2 = error_escondida_2 * d_sigmoid(self.z_2)

        error_escondida_1 = d_escondido_2.dot(self.w_2.T)
        d_escondido_1 = error_escondida_1 * d_sigmoid(self.z_1)

        # Hacer el paso del gradiente dado el calculo mediate retropropagacion
        self.w_3 += self.a_2.T.dot(d_salida) * learning_rate
        self.w_2 += self.a_1.T.dot(d_escondido_2) * learning_rate
        self.w_1 += x.T.dot(d_escondido_1) * learning_rate

        self.b_3 += np.sum(d_salida, axis=0, keepdims=True) * learning_rate
        self.b_2 += np.sum(d_escondido_2, axis=0, keepdims=True) * learning_rate
        self.b_1 += np.sum(d_escondido_1, axis=0, keepdims=True) * learning_rate

    def entrenar(self, x: NDArray, y: NDArray, epocas: int, learning_rate: float):
        for epoca in range(epocas):
            output = self.propagacion_hacia_delante(x)
            self.retropropagacion(x, y, learning_rate)
            if epoca % 100 == 0:
                print(f'Epoch {epoca}, Loss: {loss(output, y):.4f}')

    def predecir(self, x: NDArray) -> NDArray:
        return self.propagacion_hacia_delante(x)

if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Iniciar el model y entrenar en el dataset
    red = RedNeuronal(input_size=2)
    red.entrenar(x, y, epocas=10000, learning_rate=0.1)

    # Hacer predicciones
    print("Predicciones:")
    for i in x:
        print(f'{i} -> {red.predecir(i)}')

