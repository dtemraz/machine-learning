package algorithms.linear_regression.optimization.real_vector;

/**
 * This class implements stochastic gradient descent parameter optimization and can be used to train logistic regression.
 *
 * @author dtemraz
 */
public class SGDOptimizer implements Optimizer {

    private final double learningRate;
    private final int epochs;
    private final StoppingCriteria stoppingCriteria; // checked at the end of each epoch, stops learning if satisfied

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> and specified number of <em>epochs</em>.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run
     */
    public SGDOptimizer(double learningRate, int epochs) {
        this(learningRate, epochs, x -> false);
    }

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for at most <em>epochs</em> iterations.
     * Learning will stop if at the end of any epoch if <em>stoppingCriteria</em> is true.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria to exit early at the end of any epoch
     */
    public SGDOptimizer(double learningRate, int epochs, StoppingCriteria stoppingCriteria) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.stoppingCriteria = stoppingCriteria;
    }


    @Override
    public void optimize(double[][] trainingSet, double[] expected, double[] coefficients) {
        int samples = trainingSet.length;
        int epoch;
        boolean converged = false; // due to stopping criteria
        for (epoch = 0; epoch < epochs && !converged; epoch++) {
            double[] epochError = new double[samples];
            for (int sampleId = 0; sampleId < samples; sampleId++) {
                double[] sample = trainingSet[sampleId];
                double estimate = Predictor.SIGMOID.apply(sample, coefficients);
                double error = expected[sampleId] - estimate;
                epochError[sampleId] = error;
                updateCoefficients(sample, coefficients, error);
            }
            converged = stoppingCriteria.test(epochError);
        }
        System.out.printf("converged in %d epochs%n", epoch);
    }

    // adjusts coefficients with learning rate and gradient
    private void updateCoefficients(double[] input, double[] coefficients, double error) {
        double update = error * learningRate;
        // update all feature coefficients
        for (int i = 0; i < input.length; i++) {
            // ΔWk(n) = X(n) * e * η
            coefficients[i] += input[i] * update;
        }
        // update bias coefficient
        int bias = coefficients.length - 1;
        coefficients[bias] += update;
    }

}
