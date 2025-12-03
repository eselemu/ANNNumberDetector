package ai.numberdetector;

import java.util.ArrayList;
import java.util.List;

public class Brain extends Thread {
    public ANN ann = null;
    public double sumSquareError = 0;
    public double accuracy = 0;
    public int epochs;
    int nInputs;
    int nOutputs;
    int nHiddenLayers;
    int nNeuronsPerHiddenLayer;
    double alpha;

    public Brain(String name, int epochs, int nInputs, int nOutputs, int nHiddenLayers, int nNeuronsPerHiddenLayer, double alpha){
        this.nInputs = nInputs;
        this.nOutputs = nOutputs;
        this.nHiddenLayers = nHiddenLayers;
        this.nNeuronsPerHiddenLayer = nNeuronsPerHiddenLayer;
        this.epochs = epochs;
        this.alpha = alpha;
        super(name);
    }

    @Override
    public void run() {
        trainOnMNIST();
    }

    public void trainOnMNIST() {
        ann = new ANN(nInputs, nOutputs, nHiddenLayers, nNeuronsPerHiddenLayer, alpha);

        int correctPredictions = 0;
        int totalSamples = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            sumSquareError = 0;
            correctPredictions = 0;
            totalSamples = 0;

            for (int i = 0; i < 6000; i++) {
                List<Double> imagePixels = arrayToList(NumberDetector.trainImages.images.get(i));

                int label = NumberDetector.trainLabels[i];
                List<Double> oneHotLabel = toOneHot(label);

                List<Double> result = train(imagePixels, oneHotLabel);

                for (int j = 0; j < result.size(); j++) {
                    sumSquareError += Math.pow(result.get(j) - oneHotLabel.get(j), 2);
                }

                if (predictDigit(result) == label) {
                    correctPredictions++;
                }
                totalSamples++;
            }

            accuracy = (double) correctPredictions / totalSamples;

            if (epoch % 10 == 0) {  // Print progress every 10 epochs
                System.out.printf("Epoch %d [%s] - SSE: %.4f, Accuracy: %.2f%%\n",
                        epoch, getName(), sumSquareError, accuracy * 100);
            }
        }

        System.out.println("Final [" + getName() + "] - SSE: " + sumSquareError + ", Accuracy: " + (accuracy * 100) + "%");
    }

    private List<Double> arrayToList(double[] array) {
        List<Double> list = new ArrayList<>();
        for (double value : array) {
            list.add(value);
        }
        return list;
    }

    private List<Double> toOneHot(int digit) {
        List<Double> oneHot = new ArrayList<>();
        for (int i = 0; i < nOutputs; i++) {
            oneHot.add(i == digit ? 1.0 : 0.0);
        }
        return oneHot;
    }

    public int predictDigit(List<Double> outputs) {
        int predictedDigit = 0;
        double maxOutput = outputs.get(0);
        for (int i = 1; i < outputs.size(); i++) {
            if (outputs.get(i) > maxOutput) {
                maxOutput = outputs.get(i);
                predictedDigit = i;
            }
        }
        return predictedDigit;
    }

    public List<Double> train(double i1, double i2, double o) {
        List<Double> inputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>();
        inputs.add(i1);
        inputs.add(i2);
        outputs.add(o);
        return ann.go(inputs, outputs);
    }

    public List<Double> train(List<Double> inputs, List<Double> outputs) {
        return ann.go(inputs, outputs);
    }

    public List<Double> evaluate (double i1, double i2) {
        List<Double> inputs = new ArrayList<>();
        inputs.add(i1);
        inputs.add(i2);

        return ann.evaluate(inputs);
    }

    public List<Double> evaluate (List<Double> inputs) {
        return ann.evaluate(inputs);
    }

    public double evaluateOnTestSet() {
        int correct = 0;
        for (int i = 0; i < NumberDetector.testImages.images.size(); i++) {
            List<Double> imagePixels = arrayToList(NumberDetector.testImages.images.get(i));
            List<Double> result = evaluate(imagePixels);
            if (predictDigit(result) == NumberDetector.testLabels[i]) {
                correct++;
            }
        }
        return (double) correct / NumberDetector.testImages.images.size();
    }
}