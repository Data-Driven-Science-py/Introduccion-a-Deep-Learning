from collections.abc import Callable
from typing import List
from numpy.typing import NDArray
import numpy as np
from perceptron import perceptron

def red_neuronal(x: NDArray, num_capa_entrada: int, num_capa_escondida: int, num_capa_salida: int) -> NDArray:
    # Crear las capas de manera manual (recordar que inician pesos aleatorios)
    entrada: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_entrada)]
    escondida: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_escondida)]
    salida: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_salida)]

    # Evaluar de manera secuencial
    a_1 = np.concatenate([percep(x) for percep in entrada], dim = -1)
    a_2 = np.concatenate([percep(a_1) for percep in escondida], dim = -1)
    a_3 = np.concatenate([percep(a_2) for percep in salida], dim = -1)

    return a_3

from collections.abc import Callable
from typing import List
from numpy.typing import NDArray
import numpy as np
from perceptron import perceptron

def red_neuronal(x: NDArray, num_capa_entrada: int, num_capa_escondida: int, num_capa_salida: int) -> NDArray:
    # Crear las capas de manera manual (recordar que inician pesos aleatorios)
    entrada: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_entrada)]
    escondida: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_escondida)]
    salida: List[Callable[[NDArray], NDArray]] = [perceptron for _ in range(num_capa_salida)]

    # Evaluar de manera secuencial
    a_1 = np.concatenate([percep(x) for percep in entrada], axis=-1)
    a_2 = np.concatenate([percep(a_1) for percep in escondida], axis=-1)
    a_3 = np.concatenate([percep(a_2) for percep in salida], axis=-1)

    return a_3

if __name__ == '__main__':
    # Ejemplo de entrada
    x = np.random.rand(5)
    num_capa_entrada = 3
    num_capa_escondida = 4
    num_capa_salida = 2

    # Resultado de la red neuronal
    resultado = red_neuronal(x, num_capa_entrada, num_capa_escondida, num_capa_salida)
    print("Resultado:", resultado)



