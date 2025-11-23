package ai.numberdetector;

import java.util.ArrayList;
import java.util.List;

public class ANN {
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha;
    private List<Layer> layers = new ArrayList<>();

    public ANN(int nI, int nO, int nH, int nPH, double a) {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if(numHidden > 0) {
            layers.add(new Layer(numNPerHidden, numInputs));

            for(int i = 0; i < numHidden - 1; i++) {
                layers.add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.add(new Layer(numOutputs, numNPerHidden));
        } else {
            layers.add(new Layer(numOutputs, numInputs));
        }
    }

    public List<Double> go(List<Double> inputValues, List<Double> desiredOutput) {
        List<Double> inputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>();

        if(inputValues.size() != numInputs) {
            System.out.println("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new ArrayList<>(inputValues);
        for(int layerIndex = 0; layerIndex < numHidden + 1; layerIndex++) {
            if(layerIndex > 0) {
                inputs = new ArrayList<>(outputs);
            }
            outputs.clear();

            for(int neuronIndex = 0; neuronIndex < layers.get(layerIndex).numNeurons; neuronIndex++) {
                double N = 0;
                layers.get(layerIndex).neurons.get(neuronIndex).inputs.clear();

                for(int inputIndex = 0; inputIndex < layers.get(layerIndex).neurons.get(neuronIndex).numInputs; inputIndex++) {
                    layers.get(layerIndex).neurons.get(neuronIndex).inputs.add(inputs.get(inputIndex));
                    N += layers.get(layerIndex).neurons.get(neuronIndex).weights.get(inputIndex) * inputs.get(inputIndex);
                }

                N -= layers.get(layerIndex).neurons.get(neuronIndex).bias;
                layers.get(layerIndex).neurons.get(neuronIndex).output = activationFunction(N);
                outputs.add(layers.get(layerIndex).neurons.get(neuronIndex).output);
            }
        }

        updateWeights(outputs, desiredOutput);
        return outputs;
    }

    private void updateWeights(List<Double> outputs, List<Double> desiredOutput) {
        double error;
        for(int i = numHidden; i >= 0; i--) {
            for(int j = 0; j < layers.get(i).numNeurons; j++) {
                if(i == numHidden) {
                    error = desiredOutput.get(j) - outputs.get(j);
                    layers.get(i).neurons.get(j).errorGradient = outputs.get(j) * (1 - outputs.get(j)) * error;
                } else {
                    layers.get(i).neurons.get(j).errorGradient = layers.get(i).neurons.get(j).output *
                            (1 - layers.get(i).neurons.get(j).output);
                    double errorGradSum = 0;
                    for(int p = 0; p < layers.get(i + 1).numNeurons; p++) {
                        errorGradSum += layers.get(i + 1).neurons.get(p).errorGradient *
                                layers.get(i + 1).neurons.get(p).weights.get(j);
                    }
                    layers.get(i).neurons.get(j).errorGradient *= errorGradSum;
                }

                for(int k = 0; k < layers.get(i).neurons.get(j).numInputs; k++) {
                    if(i == numHidden) {
                        error = desiredOutput.get(j) - outputs.get(j);
                        layers.get(i).neurons.get(j).weights.set(k,
                                layers.get(i).neurons.get(j).weights.get(k) +
                                        alpha * layers.get(i).neurons.get(j).inputs.get(k) * error);
                    } else {
                        layers.get(i).neurons.get(j).weights.set(k,
                                layers.get(i).neurons.get(j).weights.get(k) +
                                        alpha * layers.get(i).neurons.get(j).inputs.get(k) *
                                                layers.get(i).neurons.get(j).errorGradient);
                    }
                }

                layers.get(i).neurons.get(j).bias += alpha * -1 * layers.get(i).neurons.get(j).errorGradient;
            }
        }
    }

    private double activationFunction(double value) {
        return sigmoid(value);
    }

    private double step(double value) {
        if(value < 0) return 0;
        else return 1;
    }

    private double sigmoid(double value) {
        double k = Math.exp(value);
        return k / (1.0f + k);
    }

    private double sinusoid(double value) {
        return Math.sin(value);
    }

    private double arcTan(double value) {
        return Math.atan(value);
    }

    private double softSign(double value) {
        return value / (1 + Math.abs(value));
    }

    public List<Double> evaluate(List<Double> inputValues) {
        List<Double> inputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>();

        if(inputValues.size() != numInputs) {
            System.out.println("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new ArrayList<>(inputValues);
        for(int layerIndex = 0; layerIndex < numHidden + 1; layerIndex++) {
            if(layerIndex > 0) {
                inputs = new ArrayList<>(outputs);
            }
            outputs.clear();

            for(int neuronIndex = 0; neuronIndex < layers.get(layerIndex).numNeurons; neuronIndex++) {
                double N = 0;

                // Note: We don't clear inputs here since we're not storing them for backpropagation
                for(int inputIndex = 0; inputIndex < layers.get(layerIndex).neurons.get(neuronIndex).numInputs; inputIndex++) {
                    N += layers.get(layerIndex).neurons.get(neuronIndex).weights.get(inputIndex) * inputs.get(inputIndex);
                }

                N -= layers.get(layerIndex).neurons.get(neuronIndex).bias;
                double neuronOutput = activationFunction(N);
                outputs.add(neuronOutput);
            }
        }

        return outputs;
    }
}
