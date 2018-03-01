package supervised.learning.algorithms;

import supervised.learning.samples.LearningSample;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements Delta rule learning backed by <strong>gradient descent</strong>. The implementation given in this class is
 * true gradient descent, weights are updated with cumulative error after all learning samples have been seen.
 *
 * <p>
 * For any non trivial networks this would be to slow, therefore approach given in this class is usually combined
 * with {@link DeltaRuleStochasticGradientDescent} which gives us <em>batch</em> learning.
 * </p>
 *
 * @author dtemraz
 */
public class DeltaRuleGradientDescent implements Supervisor {

    private static final double DEFAULT_LEARNING_RATE = 5.6908e-09;
    private static final double DEFAULT_ERROR_TOLERANCE = 0.02;
    private static final double MAX_EPOCH = 350_000;

    private double learningRate;
    private double errorTolerance;
    private double maxEpoch;

    public DeltaRuleGradientDescent() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_ERROR_TOLERANCE, MAX_EPOCH);
    }

    public DeltaRuleGradientDescent(double learningRate, double errorTolerance, double maxEpoch) {
        this.learningRate = learningRate;
        this.errorTolerance = errorTolerance;
        this.maxEpoch = maxEpoch;
    }

    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        int epoch;
        for (epoch = 0; epoch < maxEpoch; epoch++) {
            double[] epochError = new double[samples.size()]; // error for each sample in this epoch
            double[] delta = new double[weights.length]; // accumulate delta per weight for each sample until all learning samples seen
            for (int sample = 0; sample < samples.size(); sample++) {
                LearningSample learningSample = samples.get(sample);
                double[] input = learningSample.getInput();
                double estimated = neuronOutput.apply(input, weights);
                double error = learningSample.getDesiredOutput() - estimated;

                updateDeltaPerWeight(delta, input, error);

                epochError[sample] = error;
            }
            // early stopping criteria: global(epoch) error is less than threshold
            if (converged(epochError)) {
                break;
            }
            updateWeights(weights, delta);
        }
        System.out.println(String.format("converged in %d epoch", epoch));
    }

    // learning will converge if cumulative epoch error is less than a threshold
    private boolean converged(double[] epochError) {
        double sumSquared = 0;
        for (double e : epochError) {
            sumSquared += e * e;
        }
        return sumSquared < errorTolerance;
    }

    // accumulates error delta per each weight
    private void updateDeltaPerWeight(double[] delta, double[] input, double error) {
        for (int feature = 0; feature < delta.length; feature++) {
            delta[feature] += error * input[feature];
        }
    }

    // updates weights with accumulated error
    private void updateWeights(double[] weights, double[] accumulatedDelta) {
        for (int feature = 0; feature < weights.length; feature++) {
            weights[feature] += learningRate * accumulatedDelta[feature];
        }
    }

        /* MATLAB version vs java
      err = zeros(1, max_epoch);
      while n < max_epoch
          n = n + 1;
          e = d - w*x;
          w = w + ni * e * x';

        err(n) = sum(e.*e);
        if err(n) < errorTolerance
            break
        end
      end
     */

}