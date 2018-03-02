package neural_net;

import utilities.Vector;
import neural_net.learning.samples.LearningSample;
import neural_net.learning.algorithms.Supervisor;

import java.util.List;
import java.util.function.Function;

/**
 * The class implements model for artificial non_linear.supervised.neuron that can be trained to do simple <em>linear</em> regression
 * or binary classification.
 *
 * <p>
 * In training phase neuron will adjust weights until stopping criteria defined by supervisor has been achieved.
 * Afterwards, correct output can be calculated for inputs which have not been shown to non_linear.supervised.neuron in training phase.
 * Learning convergence is guaranteed to happen for linearly separable learning samples.
 * </p>
 *
 * The method {@link #train(List)} teaches neuron to perform linear regression or binary classification.
 * User can use method {@link #output(double[])} and {@link #output(LearningSample)} to calculate trained output for given data sample.
 *
 * @author dtemraz
 */
public class Neuron {
        
    private final Function<Double, Double> activation; // 'brain' of non_linear.supervised.neuron, function applied over weighted sum of inputs
    private final Supervisor supervisor; // algorithm to teach non_linear.supervised.neuron correct weights to solve a linearly separable problem
    private final Function<Double, Double> quantization; // optional function that can decorate output
    
    private double[] weights; // adjusted in training phase, they define coefficients for functional mapping of input
    
    public static final int BIAS_SIGNAL = 1; // we always use BIAS to be able to move decision boundary relative to origin
    
    public Neuron(int features, Function<Double, Double> activation, Supervisor supervisor) {
        this(features, activation, supervisor, Function.identity());
    }

    public Neuron(int features, Function<Double, Double> activation, Supervisor supervisor, Function<Double, Double> quantization) {
        this(Vector.randomArray(features + 1), activation, supervisor, quantization);
    }
    
    public Neuron(double[] weights, Function<Double, Double> activation, Supervisor supervisor, Function<Double, Double> quantization) {
        this.weights = weights;
        this.activation = activation;
        this.supervisor = supervisor;
        this.quantization = quantization;
    }

    /**
     * Returns estimated output for <code>data</code> vector.
     * The output is calculated as weight sum of data components(dot product) fed into {@link #activation} function.
     *
     * @param data vector that should be estimated or classified
     * @return estimated output for data
     */
    public double output(double[] data) {
        // user should not specify bias feature in data, therefore we inject bias  signal for the user
        double dotProduct = Vector.dotProduct(Vector.copyWithFirst(data, BIAS_SIGNAL), weights);
        return quantization.apply(activation.apply(dotProduct));
    }

    /**
     * Returns estimated output for learning <code>sample</code>.
     * The output is calculated as weight sum of data components(dot product) from sample input, fed into {@link #activation} function.
     *
     * @param sample that should be estimated or classified
     * @return estimated output for sample
     */
    public double output(LearningSample sample) {
        double dotProduct = Vector.dotProduct(sample.getInput(), weights); // learning sample already has bias feature
        return quantization.apply(activation.apply(dotProduct));
    }

    /**
     * Teaches the non_linear.supervised.neuron correct weights to represent input samples defined with </code>samples</code>.
     * Once this method finishes, user can apply seen or unseen input samples.
     * 
     * @param samples to train the non_linear.supervised.neuron correct weights
     */
    public void train(List<LearningSample> samples) {
        supervisor.train(samples, weights, (x, w) -> activation.apply(Vector.dotProduct(x, w)));
    }

}