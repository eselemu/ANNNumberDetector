package ai.numberdetector;

import java.util.ArrayList;
import java.util.List;

public class Brain extends Thread {
    public ANN ann = null;
    public double sumSquareError = 0;
    public int epochs;
    double step;

    public Brain(String name, int epochs, double step){
        this.epochs = epochs;
        this.step = step;
        super(name);
    }

    @Override
    public void run() {
        List<Double> result;

            ann = new ANN(2, 1, 1, 2, step);

            for (int i = 0; i < epochs; i++) {
                sumSquareError = 0;
                result = train(1, 1, 0);
                sumSquareError += Math.pow(result.get(0) - 0, 2);
                result = train(1, 0, 1);
                sumSquareError += Math.pow(result.get(0) - 1, 2);
                result = train(0, 1, 1);
                sumSquareError += Math.pow(result.get(0) - 1, 2);
                result = train(0, 0, 0);
                sumSquareError += Math.pow(result.get(0) - 0, 2);
            }

            System.out.println("SSE [" + getName() + "]: " + sumSquareError);
    }

    public List<Double> train(double i1, double i2, double o) {
        List<Double> inputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>();
        inputs.add(i1);
        inputs.add(i2);
        outputs.add(o);
        return ann.go(inputs, outputs);
    }

    public List<Double> evaluate (double i1, double i2) {
        List<Double> inputs = new ArrayList<>();
        inputs.add(i1);
        inputs.add(i2);

        return ann.evaluate(inputs);
    }
}