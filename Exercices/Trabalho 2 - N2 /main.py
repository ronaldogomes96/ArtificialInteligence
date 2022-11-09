import numpy as np
from perceptron import train_percetron, test_perceptron, train_adaline, test_adaline

## Porta nand
x_nand = np.array([ [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1] ])
y_nand = np.array([1, 1, 1, 0])

train_percetron(x_nand, y_nand)

print(test_perceptron([1, 0]))

train_percetron(x_nand, y_nand)

print(test_adaline([1, 0]))

## Porta NOR

x_nor = np.array([ [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1] ])
y_nor = np.array([1, 0, 0, 0])

train_percetron(x_nor, y_nor)

print(test_perceptron([1, 0]))

train_percetron(x_nor, y_nor)

print(test_adaline([1, 0]))