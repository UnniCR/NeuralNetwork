package com.unni.test;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class NeuralNW {
    
    public static void main(String[] args) {
        // Input training data
        double[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] output = {{0}, {1}, {1}, {1}};
        
        // Create a training dataset
        MLDataSet trainingSet = new BasicMLDataSet(input, output);
        
        // Create a neural network
        BasicNetwork neuralNetwork = new BasicNetwork();
        neuralNetwork.addLayer(new BasicLayer(null, true, 2)); // Input layer
        neuralNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), true, 4)); // Hidden layer
        neuralNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1)); // Output layer
        neuralNetwork.getStructure().finalizeStructure();
        neuralNetwork.reset();
        
        // Train the neural network
        Backpropagation train = new Backpropagation(neuralNetwork, trainingSet);
        int epoch = 0;
        do {
            train.iteration();
            epoch++;
            System.out.println("Epoch #" + epoch + " Error: " + train.getError());
        } while(train.getError() > 0.01); // Set the desired error threshold
        
        // Test the trained neural network
        System.out.println("Neural Network Results:");
        for (MLDataPair pair : trainingSet) {
            MLData outputData = neuralNetwork.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + ", " + pair.getInput().getData(1)
                    + ", Actual=" + outputData.getData(0) + ", Ideal=" + pair.getIdeal().getData(0));
        }
        
        // Shut down Encog
        Encog.getInstance().shutdown();
    }
}

