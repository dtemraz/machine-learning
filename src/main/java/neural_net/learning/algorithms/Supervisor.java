package neural_net.learning.algorithms;

import neural_net.learning.samples.LearningSample;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This interface lets user train neuron with one of the <em>non_linear.supervised</em> learning algorithms.
 * 
 * @author dtemraz
 */
public interface Supervisor {
    
    /**
     * Trains the non_linear.supervised.neuron correct weights for linear regression or binary classification of data set
     * represented with <code>samples</code>.
     *
     * <p>The method should have side-effect to adjust values of passed <em>weights</em> directly.</p>
     * 
     * @param samples to train neuron with
     * @param weights adjusted with training procedure
     * @param neuronOutput function that calculates neuron's output for given sample and weights
     */    
    void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput);
}