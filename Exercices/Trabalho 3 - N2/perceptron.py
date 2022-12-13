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

        print("\nEpoch: ", epoch)
        number_of_errors = 0
        x_data = pd.DataFrame(x)

        for index, x_row in x_data.iterrows():
            x_row = np.array(x_row)
            x_row = np.append(x_row, bias[0])

            sum = np.dot(x_row, weights)

            y_train = degree(sum)

            network_error = y[index] - y_train

            if y[index] != y_train:
                for weight_index, weight in enumerate(weights):
                    weights[weight_index] = weight + (learning_rate * network_error * x_row[weight_index])
                number_of_errors += 1

        if number_of_errors == 0:
            return


def test_perceptron(x):
    x.append(bias[0])
    sum = np.dot(x, weights)
    return degree(sum)


def degree(result):
    return 1 if result >= 0 else 0
