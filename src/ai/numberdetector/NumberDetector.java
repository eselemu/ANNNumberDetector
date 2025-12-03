package ai.numberdetector;

import ai.numberdetector.image.file.MNISTData;
import ai.numberdetector.image.file.MNISTLoader;
import ai.numberdetector.image.file.MNISTImageSaver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class NumberDetector {

    public static final String TRAINING_IMG_FILE = "dataset/train-images.idx3-ubyte";
    public static final String TRAINING_LBL_FILE = "./dataset/train-labels.idx1-ubyte";
    public static final String TEST_IMG_FILE = "./dataset/t10k-images.idx3-ubyte";
    public static final String TEST_LBL_FILE = "./dataset/t10k-labels.idx1-ubyte";

    public static final int nBrains = 3;
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

        // Start interactive testing
        interactiveTesting();
    }

    public static void interactiveTesting() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n=== MNIST DIGIT CLASSIFIER INTERACTIVE TESTING ===");
        System.out.println("Test set contains 10,000 images (indices 0-9999)");
        System.out.println("Commands:");
        System.out.println("  [0-9999] - Test specific image index");
        System.out.println("  random   - Test a random image");
        System.out.println("  first10  - Test first 10 images");
        System.out.println("  stats    - Show accuracy statistics");
        System.out.println("  exit     - Exit the program");
        System.out.println("===================================================\n");

        // Create output directory for saved images
        new File("output/test_images").mkdirs();

        while (true) {
            System.out.print("\nEnter command or image index (0-9999): ");
            String input = scanner.nextLine().trim().toLowerCase();

            if (input.equals("exit")) {
                System.out.println("Exiting interactive testing. Goodbye!");
                break;
            }
            else if (input.equals("random")) {
                testRandomImage(scanner);
            }
            else if (input.equals("first10")) {
                testFirstTenImages();
            }
            else if (input.equals("stats")) {
                showStatistics();
            }
            else {
                try {
                    int index = Integer.parseInt(input);
                    if (index >= 0 && index < testImages.images.size()) {
                        testSingleImage(index, true);
                    } else {
                        System.out.println("Invalid index. Must be between 0 and " + (testImages.images.size() - 1));
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Invalid command. Please enter a number 0-9999 or one of the commands.");
                }
            }
        }
        scanner.close();
    }

    private static void testRandomImage(Scanner scanner) {
        int randomIndex = (int) (Math.random() * testImages.images.size());
        System.out.println("Testing random image at index: " + randomIndex);
        testSingleImage(randomIndex, true);

        System.out.print("\nTest another random image? (y/n): ");
        if (scanner.nextLine().trim().toLowerCase().equals("y")) {
            testRandomImage(scanner);
        }
    }

    private static void testSingleImage(int index, boolean saveImage) {
        try {
            double[] pixels = testImages.images.get(index);
            int actualLabel = testLabels[index];

            List<Double> imagePixels = arrayToList(pixels);
            List<Double> result = brain.evaluate(imagePixels);
            int predicted = brain.predictDigit(result);

            List<Double> confidences = new ArrayList<>();
            for (int i = 0; i < result.size(); i++) {
                confidences.add(result.get(i));
            }

            System.out.println("\n=== IMAGE " + index + " ===");
            System.out.println("Actual digit: " + actualLabel);
            System.out.println("Predicted digit: " + predicted);
            System.out.println(predicted == actualLabel ? "✓ CORRECT" : "✗ WRONG");

            System.out.println("\nConfidence scores:");
            for (int i = 0; i < confidences.size(); i++) {
                System.out.printf("  %d: %.4f%s\n", i, confidences.get(i),
                        i == predicted ? " ← PREDICTED" : i == actualLabel ? " ← ACTUAL" : "");
            }

            if (saveImage) {
                String filename = String.format("output/test_images/image_%04d_label_%d_pred_%d.bmp",
                        index, actualLabel, predicted);
                MNISTImageSaver.saveImageAsBMP(pixels, testImages.cols, testImages.rows, filename, actualLabel);
                System.out.println("Image saved to: " + filename);
            }

        } catch (IOException e) {
            System.out.println("Error saving image: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("Error testing image: " + e.getMessage());
        }
    }

    private static void testFirstTenImages() {
        System.out.println("\n=== TESTING FIRST 10 IMAGES ===");
        int correct = 0;
        for (int i = 0; i < 10 && i < testImages.images.size(); i++) {
            List<Double> imagePixels = arrayToList(testImages.images.get(i));
            List<Double> result = brain.evaluate(imagePixels);
            int predicted = brain.predictDigit(result);
            int actual = testLabels[i];

            String status = predicted == actual ? "✓" : "✗";
            if (predicted == actual) correct++;

            System.out.printf("Image %d: Predicted %d, Actual %d %s\n",
                    i, predicted, actual, status);
        }
        System.out.printf("Accuracy: %d/%d = %.1f%%\n", correct, 10, (correct/10.0)*100);
    }

    private static void showStatistics() {
        if (brain == null) {
            System.out.println("No brain selected for testing.");
            return;
        }

        double testAccuracy = brain.evaluateOnTestSet();
        System.out.println("\n=== STATISTICS ===");
        System.out.println("Selected brain: " + brain.getName());
        System.out.printf("Test accuracy: %.2f%%\n", testAccuracy * 100);
        System.out.println("Test set size: " + testImages.images.size() + " images");
        System.out.println("Training set size: " + trainImages.images.size() + " images");
        System.out.println("Network architecture: " + brain.ann.numInputs + "-" +
                brain.ann.numNPerHidden + "-" + brain.ann.numOutputs);
        System.out.println("Learning rate (alpha): " + brain.ann.alpha);
    }

    public static void testSamplePredictions() {
        System.out.println("\n=== SAMPLE PREDICTIONS (First 10 images) ===");
        for (int i = 0; i < 10; i++) {
            List<Double> imagePixels = arrayToList(testImages.images.get(i));
            List<Double> result = brain.evaluate(imagePixels);
            int predicted = brain.predictDigit(result);
            int actual = testLabels[i];
            System.out.printf("Image %d: Predicted %d, Actual %d %s\n",
                    i, predicted, actual, predicted == actual ? "✓" : "✗");
        }
    }

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

            System.out.println("Dataset loaded successfully!");
            System.out.println("Training images: " + trainImages.images.size());
            System.out.println("Test images: " + testImages.images.size());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void trainMNISTNetworks() {
        double bestAccuracy = 0;
        brains = new Brain[nBrains];

        for(int b = 0; b < nBrains; b++){
            brains[b] = new Brain("Brain " + b, 50, 784, 10, 1, 128, 0.1);
            brains[b].start();
        }

        for(int b = 0; b < nBrains; b++){
            try {
                brains[b].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

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