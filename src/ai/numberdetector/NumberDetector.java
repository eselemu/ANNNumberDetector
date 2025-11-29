package ai.numberdetector;

import ai.numberdetector.image.file.MNISTData;
import ai.numberdetector.image.file.MNISTLoader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NumberDetector {

    public static final String TRAINING_IMG_FILE = "dataset/train-images.idx3-ubyte";
    public static final String TRAINING_LBL_FILE = "./dataset/train-labels.idx1-ubyte";
    public static final String TEST_IMG_FILE = "./dataset/t10k-images.idx3-ubyte";
    public static final String TEST_LBL_FILE = "./dataset/t10k-labels.idx1-ubyte";

    public static final int nBrains = 5;
    public static final int epochs = 10;
    public static Brain[] brains;
    public static Brain brain = null;

    public static MNISTData trainImages;
    public static MNISTData testImages;
    public static int[] trainLabels;
    public static int[] testLabels;

    public static void main(String[] args){
        loadDataset();
        trainMNISTNetworks();
    }

    public static void trainMNISTNetworks() {
        double bestAccuracy = 0;
        brains = new Brain[nBrains];

        // MNIST architecture: 784 inputs, 10 outputs, 1 hidden layer, 128 neurons, 0.1 alpha
        for(int b = 0; b < nBrains; b++){
            brains[b] = new Brain("Brain " + b, 50, 784, 10, 1, 128, 0.1);
            brains[b].start();
        }

        // Wait for all brains to finish training
        for(int b = 0; b < nBrains; b++){
            try {
                brains[b].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        // Select best brain based on accuracy
        for(int b = 0; b < nBrains; b++){
            double testAccuracy = brains[b].evaluateOnTestSet();
            System.out.println("Brain " + b + " Test Accuracy: " + (testAccuracy * 100) + "%");

            if (testAccuracy > bestAccuracy) {
                bestAccuracy = testAccuracy;
                brain = brains[b];
            }
        }

        System.out.println("Selected Brain: " + brain.getName());
        System.out.println("Test Accuracy: " + (bestAccuracy * 100) + "%");

        // Test on some sample images
        testSamplePredictions();
    }

    public static void testSamplePredictions() {
        System.out.println("\nSample Predictions:");
        for (int i = 0; i < 10; i++) {
            List<Double> imagePixels = arrayToList(testImages.images.get(i));
            List<Double> result = brain.evaluate(imagePixels);
            int predicted = brain.predictDigit(result);
            int actual = testLabels[i];
            System.out.printf("Image %d: Predicted %d, Actual %d %s\n",
                    i, predicted, actual, predicted == actual ? "✓" : "✗");
        }
    }

    // Helper method
    private static List<Double> arrayToList(double[] array) {
        List<Double> list = new ArrayList<>();
        for (double value : array) {
            list.add(value);
        }
        return list;
    }

    public static void loadDataset() {
        try {
            trainImages = MNISTLoader.loadImages(TRAINING_IMG_FILE);
            trainLabels = MNISTLoader.loadLabels(TRAINING_LBL_FILE);

            testImages = MNISTLoader.loadImages(TEST_IMG_FILE);
            testLabels = MNISTLoader.loadLabels(TEST_LBL_FILE);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void runXOR(){
        double tempSSE = Double.MAX_VALUE;
        brains = new Brain[nBrains];
        for(int b = 0; b < nBrains; b++){
            brains[b] = new Brain("Brain " + b, epochs, 2, 1, 1, 2, 0.4);
            brains[b].start();
            try {
                brains[b].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        for(int b = 0; b < nBrains; b++){
            if(brains[b].sumSquareError < tempSSE){
                tempSSE = brains[b].sumSquareError;
                brain = brains[b];
            }
        }

        System.out.println("Selected Brain: " + brain.getName());
        System.out.println("SSE: " + brain.sumSquareError);

        List<Double> result;
        result = brain.evaluate(1, 1);
        System.out.println(" 1 1 " + result.get(0));
        result = brain.evaluate(1, 0);
        System.out.println(" 1 0 " + result.get(0));
        result = brain.evaluate(0, 1);
        System.out.println(" 0 1 " + result.get(0));
        result = brain.evaluate(0, 0);
        System.out.println(" 0 0 " + result.get(0));
    }
}
