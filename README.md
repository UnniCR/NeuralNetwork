# Java Neural Network with Encog

This project demonstrates a simple feedforward neural network built using the [Encog Machine Learning Framework](http://www.heatonresearch.com/encog/) in Java. It is trained with backpropagation to simulate the logical **OR** function.

src/
└── com/
    └── unni/
        └── test/
            └── NeuralNW.java


## Features

- Implements a 3-layer neural network:
  - **Input Layer**: 2 neurons
  - **Hidden Layer**: 4 neurons (Sigmoid activation)
  - **Output Layer**: 1 neuron (Sigmoid activation)
- Uses **Backpropagation** training
- Trains on the OR logic gate dataset:  
  `(0 OR 0 = 0, 0 OR 1 = 1, 1 OR 0 = 1, 1 OR 1 = 1)`
- Displays training error and final prediction outputs

---

## Technologies Used

- Java
- [Encog](https://github.com/encog/encog-java-core) ML Library
- Backpropagation Algorithm
- Sigmoid Activation Function

---

## Sample Output
Neural Network Results:
0.0, 0.0, Actual=0.02, Ideal=0.0
0.0, 1.0, Actual=0.98, Ideal=1.0
1.0, 0.0, Actual=0.97, Ideal=1.0
1.0, 1.0, Actual=0.99, Ideal=1.0
