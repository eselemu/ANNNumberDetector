package ai.numberdetector;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    public int numNeurons;
    public List<Neuron> neurons = new ArrayList<>();

    public Layer(int nNeurons, int numNeuronInputs) {
        numNeurons = nNeurons;
        for(int i = 0; i < nNeurons; i++) {
            neurons.add(new Neuron(numNeuronInputs));
        }
    }
}
