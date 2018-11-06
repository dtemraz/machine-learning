package algorithms.neural_net.learning.algorithms;

import algorithms.neural_net.Activation;
import algorithms.neural_net.learning.samples.LearningSample;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements Delta rule learning backed by <strong>stochastic gradient descent</strong>.
 * The Delta rule learning attempts to minimize difference between the expected output and the output neuron calculates.
 * To achieve this we define <strong>cost function </strong>to quantify the difference, given that cost function tells us
 * how bad our neuron(network) performs.
 * Cost function is a function of weights, therefore with gradient descent we shall adjust the weights to minimize the function.
 *
 * This algorithm is specialized version of <em>back propagation</em>, it only works for single layer networks.
 *
 * <p>The implementation given in this class is <em>specialized</em> and will work only with:</p>
 * <ul>
 *     <li>linear activation, {@link Activation#IDENTITY} with MSE cost function</li>
 *     <li>unipolar sigmoid, {@link Activation#SIGMOID} with CROSS ENTROPY cost function</li>
 * </ul>
 *
 *
 * This implementation provides online(stochastic) non_linear.supervised learning model which applies update after each learning sample.
 * The update rule for the special cases defied in paragraph above is:
 *
 * ΔWk(n) = e(n) * η * x(n)
 * <ul>
 *  <li> e(n) = difference between desired output and predicted output</li>
 *  <li> η = {@link #learningRate}</li>
 *  <li> x(n) = input vector instance </li>
 * </ul>
 *
 * This is a very bad way of doing the updates in multi layer network since we would just jump all over cost function. Ideally,
 * we would use batch learning.
 *
 * <p>See also {@link DeltaRuleGradientDescent} as alternative implementation where weights are updated after all samples.</p>
 *
 * @author dtemraz
 */
public class DeltaRuleStochasticGradientDescent implements Supervisor {

    private static final int MAX_EPOCH = 100_000;
    private static final double DEFAULT_LEARNING_RATE = 0.002;
    private static final double DEFAULT_ERROR_TOLERANCE = 0.000009;

    private final double learningRate; // smaller value - more stable but slower convergence
    private final double errorTolerance; // max error in an epoch we tolerate
    private final int maxEpoch; // max number of epochs we allow before terminating learning

    public DeltaRuleStochasticGradientDescent() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_ERROR_TOLERANCE, MAX_EPOCH);
    }

    public DeltaRuleStochasticGradientDescent(double learningRate, double errorTolerance, int maxEpoch) {
        this.learningRate = learningRate;
        this.errorTolerance = errorTolerance;
        this.maxEpoch = maxEpoch;
    }

    // we could have simply proxied this method to GradientDescent#stochastic
    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        int epoch;
        for (epoch = 0; epoch < maxEpoch; epoch++) {
            boolean converged = true; // converges if all learning samples correctly classified
            for (LearningSample sample : samples) {
                double[] input = sample.getInput();
                double estimated = neuronOutput.apply(input, weights);
                double error = sample.getDesiredOutput() - estimated;
                updateWeights(input, weights, error); // here we update weights after each sample
                converged &= error < errorTolerance; // true if all samples in epoch bellow errorTolerance
            }

            if (converged) {
                break;
            }
        }
        System.out.println(String.format("converged in %d epoch", epoch));
    }

    // updates weights according to delta learning rule: ΔWk(n) = e(n) * η * x(n)
    private void updateWeights(double[] input, double[] weights, double error) {
        for (int feature = 0; feature < weights.length; feature++) {
            weights[feature] += learningRate * error * input[feature];
        }
    }

}
