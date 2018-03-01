package supervised.learning.algorithms;

import supervised.learning.samples.LearningSample;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This interface lets user train neuron with one of the <em>supervised</em> learning algorithms.
 * 
 * @author dtemraz
 */
public interface Supervisor {
    
    /**
     * Trains the supervised.neuron correct weights for linear regression or binary classification of data set
     * represented with <code>samples</code>.
     * <p>The method should have side-effect to adjust values of passed <em>weights</em> directly.</p>
     * 
     * @param samples to train neuron with
     * @param weights adjusted with training procedure
     * @param neuronOutput function that calculates supervised.neuron's output for given sample and weights
     */    
    void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput);
}