package ai.numberdetector;

import java.util.List;

public class NumberDetector {

    public static int nBrains = 5;
    public static int epochs = 10000;
    public static double step = 0.4;
    public static Brain[] brains;

    public static Brain brain = null;

    public static void main(String[] args){
        double tempSSE = Double.MAX_VALUE;
        brains = new Brain[nBrains];
        for(int b = 0; b < nBrains; b++){
            brains[b] = new Brain("Brain " + b, epochs, step);
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
