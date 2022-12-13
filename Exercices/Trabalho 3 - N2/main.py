from perceptron import train_percetron, test_perceptron
from adaline import train_adaline, test_adaline
import numpy as np
from sklearn.neural_network import MLPClassifier


def main_perceptron():
    # Port NAND
    x_nand = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    y_nand = np.array([1, 1, 1, 0])

    train_percetron(x_nand, y_nand)

    print("\nResult:", test_perceptron([1, 0]))


def main_adaline():
    # Port AND
    x_nand = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    y_nand = np.array([0, 0, 0, 1])

    train_adaline(x_nand, y_nand)

    print("\nResult:", test_adaline([0, 1]))


def main_MLP():
    # Port NOR
    x_nor = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    y_nor = np.array([1, 0, 0, 0])

    mlp = MLPClassifier(verbose=True)

    mlp.fit(x_nor, y_nor)

    predict = np.array([1, 1]).reshape(1, -1)

    print("Result: ", mlp.predict(predict)[0])


if __name__ == "__main__":
    main_MLP()
