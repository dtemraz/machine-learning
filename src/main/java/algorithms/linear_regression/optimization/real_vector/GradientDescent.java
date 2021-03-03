package algorithms.linear_regression.optimization.real_vector;

import structures.RandomizedQueue;
import utilities.math.Vector;

/**
 * This class implements gradient descent method for algorithms.linear_regression.optimization of theta.
 * This is not a general purpose gradient descent because it makes some assumptions, namely:
 * <ul>
 *   <li>least squares cost function with linear activation</li>
 *   <li>cross entropy cost function with sigmoid activation</li>
 * </ul>
 *
 * These assumptions simplify calculation a little bit because they lead to a very simple coefficient update rule:
 *  <p>ΔWk(n) = e(n) * η * x(n)</p>
 *
 * <p>The class offers three methods to update theta/weights:</p>
 * <ul>
 *     <li>{@link #stochastic(double[][], double[], double[], Predictor)} which makes update after each sample</li>
 *     <li>{@link #batch(double[][], double[], double[], Predictor)} which makes update after all samples are seen</li>
 *     <li>{@link #miniBatch(double[][], double[], double[], int, Predictor)} which makes update after batch size samples</li>
 * </ul>
 *
 * @author dtemraz
 */
public class GradientDescent {

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // number of epochs this algorithm will run, unless stopping criteria stops the algorithm early
    private final StoppingCriteria stoppingCriteria; // checked at the end of each epoch, stops learning if satisfied

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> and specified number of <em>epochs</em>. This
     * instance will not use any stopping criteria.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run
     */
    public GradientDescent(double learningRate, int epochs) {
        this(learningRate, epochs, x -> false);
    }

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if at the end of any epoch <em>stoppingCriteria</em> is true.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria to exit early at the end of any epoch
     */
    public GradientDescent(double learningRate, int epochs, StoppingCriteria stoppingCriteria) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.stoppingCriteria = stoppingCriteria;
    }

    /**
     * Runs stochastic gradient descent to optimize <em>theta</em>. This means that theta will be updated
     * after each sample.
     *
     * @param in matrix of input samples, each row defines a single sample
     * @param out vector of values associated with samples
     * @param coefficients to optimize
     * @param predictor function that calculates value from the sample and <em>theta</em>
     */
    public void stochastic(double[][] in, double[] out, double[] coefficients, Predictor predictor) {
        int samples = in.length; // number of learning samples
        int epoch; // save value so we print it when algorithm finishes
        boolean converged = false; // this will be true if the algorithm converges early due to stopping criteria
        for (epoch = 0; epoch < epochs && !converged; epoch++) {
            double[] epochError = new double[samples];
            for (int sample = 0; sample < samples; sample++) {
                double[] input = in[sample];
                double error = out[sample] - predictor.apply(input, coefficients);
                epochError[sample] = error; // stopping criteria requires error for each sample in an epoch
                updateCoefficients(input, coefficients, error * learningRate);
            }
            converged = stoppingCriteria.test(epochError);
        }
        System.out.println(String.format("converged in %d epochs", epoch));
    }

    /**
     * Runs batch gradient descent to optimize <em>theta</em>. This means that theta will be updated
     * at the end of each epoch after all samples have been seen.
     *
     * @param in matrix of input samples, each row defines a single sample
     * @param out vector of values associated with samples
     * @param coefficients to optimize
     * @param predictor function that calculates value from the sample and <em>theta</em>
     */
    public void batch(double[][] in, double[] out, double[] coefficients, Predictor predictor) {
        miniBatch(in, out, coefficients, in.length, predictor);
    }

    /**
     * Runs mini batch gradient descent to optimize <em>theta</em>. This means that theta will be updated
     * in batches after each <em>batchSize</em> samples are seen.
     *
     * @param in matrix of input samples, each row defines a single sample
     * @param out vector of values associated with samples
     * @param coefficients to optimize
     * @param batchSize number of samples in a batch over which to compute gradient
     * @param predictor function that calculates value from the sample and <em>theta</em>
     */
    public void miniBatch(double[][] in, double[] out, double[] coefficients, int batchSize, Predictor predictor) {
        batchCoefficients(in, out, coefficients, batchSize, predictor);
    }

    // performs batch optimization of theta, where batch size <= sample.size
    private void batchCoefficients(double[][] input, double[] out, double[] coefficients, int batchSize, Predictor predictor) {
        batchSize = Math.min(batchSize, input.length); // we cannot have batch size greater than number of available samples
        int features = input[0].length; // all samples will have same number of features
        boolean converged = false; // this will be true if the algorithm converges early due to stopping criteria
        int epoch;  // save value so we print it when algorithm finishes
        double updateFactor = learningRate / batchSize; // we will update by weights by average of gradient
        RandomizedQueue<Integer> dataRows = RandomizedQueue.intQueue(input.length); // random samples iteration order

        for (epoch = 0; epoch < epochs && !converged; epoch++) {
            double[] gradient = new double[features]; // change of weights is proportional to gradient
            double[] epochError = new double[input.length]; // vector of error per sample in this epoch
            double batchError = 0;
            int sample = 0; // sample count in this epoch
            int batchCount = 0; // count of samples in this batch
            // iterator guarantees random order in each epoch
            for (int row : dataRows) {
                sample++;
                batchCount++;
                double estimate = predictor.apply(input[row], coefficients);
                double error = out[row] - estimate;
                epochError[sample] = error;

                // collect gradient for feature weights until the end of batch
                Vector.mergeSum(gradient, Vector.multiply(input[row], error));
                // for bias term, gradient is equal to the error
                batchError += error;

                if (batchCount == batchSize || dataRows.isEmpty()) {
                    batchUpdate(gradient, coefficients, batchSize, batchError);
                    gradient = new double[features];
                    batchCount = 0;
                    batchError = 0;
                }
            }
            converged = stoppingCriteria.test(epochError);
        }
        System.out.println(String.format("converged in %d epochs", epoch));
    }

    private void updateCoefficients(double[] input, double[] coefficients, double update) {
        // update all feature coefficients
        for (int i = 0; i < input.length; i++) {
            // ΔWk(n) = e(n) * η * X(n) , sum existing theta with this delta
            coefficients[i] += input[i] * update;
        }
        // update bias coefficient
        int bias = coefficients.length - 1;
        coefficients[bias] += update;
    }

    private void batchUpdate(double[] gradient, double[] coefficients, double batchSize, double batchError) {
        double batchUpdateFactor = learningRate / batchSize;
        // update all feature coefficients
        for (int i = 0; i < gradient.length; i++) {
            // ΔWk(n) = e(n) * η * X(n) , sum existing theta with this delta
            coefficients[i] += gradient[i] * batchUpdateFactor;
        }
        // update bias coefficient
        int bias = coefficients.length - 1;
        coefficients[bias] += batchError * batchUpdateFactor;
    }

}