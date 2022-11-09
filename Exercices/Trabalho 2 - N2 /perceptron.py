import numpy as np
import pandas as pd

weights = []
bias = np.array([-1])


def train_percetron(x,
                    y,
                    number_of_epochs=1000,
                    learning_rate=0.1):
    for index in range(len(x[0]) + 1):
        weights.append(0.5)

    x = pd.DataFrame(x)
    number_of_erros = 1

    for epoch in range(number_of_epochs):
        print(epoch)
        if number_of_erros == 0:
            return

        number_of_erros = 0

        for index, x_row in x.iterrows():
            x_row = np.array(x_row)
            x_row = np.append(x_row, bias[0])

            sum = np.dot(x_row, weights)

            y_train = 1 if sum >= 0 else 0

            network_error = y[index] - y_train

            if network_error != 0:
                for weight_index, weight in enumerate(weights):
                    weights[weight_index] = weight + (learning_rate * network_error * x_row[weight_index])
                bias[0] = bias[0] + learning_rate * network_error
                number_of_erros += 1


def test_perceptron(x):
    x.append(bias[0])
    sum = np.dot(x, weights)
    return 1 if sum >= 0 else 0


def train_adaline(x,
                  y,
                  number_of_epochs=1000,
                  learning_rate=0.1,
                  max_error=0):
    for index in range(len(x[0]) + 1):
        weights.append(0.5)

    x = pd.DataFrame(x)

    for epoch in range(number_of_epochs):
        last_mean_square_error = calcule_mean_square(x, y)

        for index, x_row in x.iterrows():
            x_row = np.array(x_row)
            x_row = np.append(x_row, bias[0])

            y_train = np.dot(x_row, weights)

            network_error = y[index] - y_train

            for weight_index, weight in enumerate(weights):
                weights[weight_index] = weight + (learning_rate * network_error * x_row[weight_index])
            bias[0] = bias[0] + learning_rate * network_error

        actual_mean_square_error = calcule_mean_square(x, y)

        if abs(actual_mean_square_error - last_mean_square_error) <= max_error:
            return


def calcule_mean_square(x, y):
    mean_square_error = 0

    for index, x_row in x.iterrows():
        x_row = np.array(x_row)
        x_row = np.append(x_row, bias[0])

        result = np.dot(x_row, weights)

        mean_square_error += ((y[index] - result) ** 2)

    return mean_square_error / (len(x[0]))


def test_adaline(x):
    x.append(bias[0])
    sum = np.dot(x, weights)
    return 1 if sum >= 0 else 0
