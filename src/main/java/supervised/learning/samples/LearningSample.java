package supervised.learning.samples;

import supervised.learning.VectorUtils;
import supervised.neuron.Neuron;

import java.util.Arrays;

/**
 * This class models a single instance of learning sample used to train {@link Neuron}.
 * The class ensures that input is extended with {@link Neuron#BIAS_SIGNAL} so the client of the API doesn't need to think about it.
 * 
 * @author dtemraz
 */
public class LearningSample {
    
    private final double[] input; // input vector with bias in 0th spot
    private final double desiredOutput; // desired output for input vector
        
    public LearningSample(double[] input, double output) {
        // inject BIAS_SIGNAL into 0th spot of input since bias should always be active in supervised.learning
        this.input = VectorUtils.copyWithFirst(input, Neuron.BIAS_SIGNAL);
        this.desiredOutput = output;
    }

    /**
     * Returns an input vector, modified with {@link Neuron#BIAS_SIGNAL} which is injected in 0th spot of the array.
     *
     * @return an input vector, modified with {@link Neuron#BIAS_SIGNAL} which is injected in 0th spot of the array
     */
    public double[] getInput() {
        return input;
    }

    /**
     * Returns desired output for given <em>input</em>.
     *
     * @return desired output for given <em>input</em>
     */
    public double getDesiredOutput() {
        return desiredOutput;
    }


    @Override
    public String toString() {
        return "input: " + Arrays.toString(input) + " , desired output: " + desiredOutput;
    }

}
