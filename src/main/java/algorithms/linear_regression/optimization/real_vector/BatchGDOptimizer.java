package algorithms.linear_regression.optimization.real_vector;

import structures.RandomizedQueue;
import utilities.math.Vector;

/**
 * This class implements batch gradient descent parameter optimization and can be used to train logistic regression.
 * If a user specifies {@link BatchGDOptimizer#batchSize} as a number less than dataset size, the algorithm will be mini batch.
 *
 * @author dtemraz
 */
public class BatchGDOptimizer implements Optimizer {

    private final double learningRate;
    private final int epochs;
    private final int batchSize;
    private final StoppingCriteria stoppingCriteria; // checked at the end of each epoch, stops learning if satisfied

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> and specified number of <em>epochs</em>.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run
     * @param batchSize samples size to compute(approximate) gradient
     */
    public BatchGDOptimizer(double learningRate, int epochs, int batchSize) {
        this(learningRate, epochs, batchSize, x -> false);
    }

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>.
     * Learning will stop if at the end of any epoch if <em>stoppingCriteria</em> is true.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param batchSize samples size to compute(approximate) gradient
     * @param stoppingCriteria to exit early at the end of any epoch
     */
    public BatchGDOptimizer(double learningRate, int epochs, int batchSize, StoppingCriteria stoppingCriteria) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.stoppingCriteria = stoppingCriteria;
    }

    @Override
    public void optimize(double[][] trainingSet, double[] expected, double[] coefficients) {
        int samples = trainingSet.length;
        if (batchSize > samples) {
            throw new IllegalArgumentException(String.format("batch size: %d greater than number of samples: %d", batchSize, samples));
        }

        int features = trainingSet[0].length;
        double[] gradient = new double[features]; // reset on each batch update

        boolean converged = false; // true if GD converges early due to stopping criteria
        int epoch;
        RandomizedQueue<Integer> sampleIndexes = RandomizedQueue.intQueue(samples); // iterator gives random order each time
        for (epoch = 0; epoch < epochs && !converged; epoch++) {
            double[] epochError = new double[samples];
            double batchError = 0;
            int sample = 0;
            int batchCount = 0;

            // ensures random samples in each epoch
            for (int sampleId : sampleIndexes) {
                double estimate = Predictor.SIGMOID.apply(trainingSet[sampleId], coefficients);
                double error = expected[sampleId] - estimate;
                epochError[sample++] = error;
                batchError += error;  // for bias term, gradient is equal to the error
                Vector.mergeSum(gradient, Vector.multiply(trainingSet[sampleId], error));

                // weights should be updated with average gradient in the batch
                if (++batchCount == batchSize || sampleIndexes.isEmpty()) {
                    updateCoefficients(gradient, coefficients, batchSize, batchError);
                    gradient = new double[features];
                    batchCount = 0;
                    batchError = 0;
                }
            }
            converged = stoppingCriteria.test(epochError);
        }
        System.out.println(String.format("converged in %d epochs", epoch));
    }

    // adjusts coefficients with learning rate and average gradient in batch
    private void updateCoefficients(double[] gradient, double[] coefficients, double batchSize, double batchError) {
        double batchUpdateFactor = learningRate / batchSize;
        // update all feature coefficients
        for (int i = 0; i < gradient.length; i++) {
            // ΔWk(n) = (η/m) * avg(X*e)
            coefficients[i] += gradient[i] * batchUpdateFactor;
        }
        // update bias coefficient, gradient is the (average) error for bias
        int bias = coefficients.length - 1;
        coefficients[bias] += batchError * batchUpdateFactor;
    }

}
