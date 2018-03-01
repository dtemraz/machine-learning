package supervised.learning.algorithms;

import supervised.learning.samples.LearningSample;

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
 * <p>
 * The implementation given in this class is <em>specialized</em> and will work only with:
 * <ul>
 *     <li>linear activation, {@link supervised.neuron.Activation#IDENTITY} with MSE cost function</li>
 *     <li>unipolar sigmoid, {@link supervised.neuron.Activation#SIGMOID} with CROSS ENTROPY cost function</li>
 * </ul>
 * </p>
 *
 * This implementation provides online(stochastic) supervised learning model which applies update after each learning sample.
 * The update rule for the special cases defied in paragraph above is:
 *
 * ΔWk(n) = e(n) * η * x(n)
 * <ul>
 *  <li> e(n) = difference between desired output and predicted output</li>
 *  <li> η = {@link #learningRate}</li>
 *  <li> x(n) = input vector instance </li>
 * </ul>
 *
 * This is a very bad way of doing the updates in multi layer network since we would just jump all over cost function.
 *
 * @author dtemraz
 */
public class DeltaRuleStochasticGradientDescent implements Supervisor {

    private static final int MAX_EPOCH = 100_000;
    private static final double DEFAULT_LEARNING_RATE = 0.002;
    private static final double DEFAULT_ERROR_TOLERANCE = 0.000009;

    private final double learningRate;
    private final double errorTolerance;
    private final double maxEpoch;

    public DeltaRuleStochasticGradientDescent() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_ERROR_TOLERANCE, MAX_EPOCH);
    }

    public DeltaRuleStochasticGradientDescent(double learningRate, double errorTolerance, double maxEpoch) {
        this.learningRate = learningRate;
        this.errorTolerance = errorTolerance;
        this.maxEpoch = maxEpoch;
    }

    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        int epoch;
        for (epoch = 0; epoch < maxEpoch; epoch++) {
            boolean converged = true; // converges if all learning samples correctly classified
            for (LearningSample sample : samples) {
                double[] input = sample.getInput();
                double estimated = neuronOutput.apply(input, weights);
                double error = sample.getDesiredOutput() - estimated;
                updateWeights(input, weights, error);
                converged &= error < errorTolerance;
            }

            if (converged) {
                break;
            }
        }
        System.out.println(String.format("converged in %d epoch", epoch));
    }

    private void updateWeights(double[] input, double[] weights, double error) {
        for (int feature = 0; feature < weights.length; feature++) {
            weights[feature] += learningRate * error * input[feature];
        }
    }

}
