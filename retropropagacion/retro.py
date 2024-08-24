import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) ## Otra funcion no linear de activacion

def d_sigmoid(x):
    return x * (1 - x) ## Derivada de la funcion de activacion

class RedNeuronal:
    # Constructor de la red neuronal
    def __init__(self, input_size: int):
        self.w_1 = np.random.randint(input_size, 3)
        self.w_2 = np.random.randint(3, 4)
        self.w_3 = np.random.randint(4, 2)

        self.weights_input_hidden = np.random.randn(input_size, hidden_size) #Inicia los parameteros de la primera capa
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) # Inicia los parametros de la ultima capa
        self.bias_hidden = np.zeros((1, hidden_size)) #Inicia la constante de la primera capa
        self.bias_output = np.zeros((1, output_size)) #Incia la constante de la ultima capa

    def feedforward(self, x):
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return output

    def backpropagation(self, x, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Calculate hidden layer error and gradients
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(x)
            self.backpropagation(x, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, x):
        return self.feedforward(x)

# Example usage
if __name__ == '__main__':
    # XOR dataset
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize and train the network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(x, y, epochs=10000, learning_rate=0.1)

    # Make predictions
    print("Predictions:")
    for i in x:
        print(f'{i} -> {nn.predict(i)}')

