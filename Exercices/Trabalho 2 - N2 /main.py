import numpy as np
from perceptron import train_percetron, test_perceptron
from adaline import train_adaline, test_adaline

## Porta nand
x_nand = np.array([ [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1] ])
y_nand = np.array([1, 1, 1, 0])

train_adaline(x_nand, y_nand)
train_percetron(x_nand, y_nand)

print(test_adaline([0, 0]))
print(test_perceptron([0, 0]))
