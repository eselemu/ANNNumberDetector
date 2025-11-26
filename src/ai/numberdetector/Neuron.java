package ai.numberdetector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    public int numInputs;
    public double bias;
    public double output;
    public double errorGradient;
    public List<Double> weights = new ArrayList<>();
    public List<Double> inputs = new ArrayList<>();
    private Random random = new Random();

    public Neuron(int nInputs) {
        double scale = Math.sqrt(2.0 / nInputs);
        random = new Random();

        bias = random.nextGaussian() * scale;
        numInputs = nInputs;
        for(int i = 0; i < nInputs; i++) {
            weights.add(random.nextGaussian() * scale);
        }
    }
}
