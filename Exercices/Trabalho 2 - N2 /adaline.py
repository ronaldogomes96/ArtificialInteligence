import numpy as np
import pandas as pd


weights = []
bias = np.array([-1])


def train_adaline(x,
                  y,
                  number_of_epochs=1000,
                  learning_rate=0.5,
                  max_error=0.5):
    for index in range(len(x[0]) + 1):
        weights.append(0.5)

    x = pd.DataFrame(x)

    for epoch in range(number_of_epochs):
        print(epoch)
        last_mean_square_error = calcule_mean_square(x, y)

        for index, x_row in x.iterrows():
            x_row = np.array(x_row)
            x_row = np.append(x_row, bias[0])

            y_train = np.dot(x_row, weights)

            network_error = y[index] - y_train

            for weight_index, weight in enumerate(weights):
                weights[weight_index] = weight + (learning_rate * network_error * x_row[weight_index])

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