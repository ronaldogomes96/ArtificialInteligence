import numpy as np
import pandas as pd

weights = []
bias = np.array([-1])


def train_percetron(x,
                    y,
                    number_of_epochs=1000,
                    learning_rate=0.3):
    for index in range(len(x[0]) + 1):
        weights.append(0.5)

    for epoch in range(number_of_epochs):

        print(epoch)
        number_of_erros = 0
        new_X = pd.DataFrame(x)

        for index, x_row in new_X.iterrows():
            x_row = np.array(x_row)
            x_row = np.append(x_row, bias[0])

            sum = np.dot(x_row, weights)

            y_train = 1 if sum >= 0 else 0

            network_error = y[index] - y_train

            if y[index] != y_train:
                for weight_index, weight in enumerate(weights):
                    weights[weight_index] = weight + (learning_rate * network_error * x_row[weight_index])
                number_of_erros += 1

        if number_of_erros == 0:
            return

def test_perceptron(x):
    x.append(bias[0])
    sum = np.dot(x, weights)
    return 1 if sum >= 0 else 0
