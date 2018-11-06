package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.Activation;
import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.List;
import java.util.Map;

/**
 * This class implements gradient descent optimization for sparse text classification with logistic regression.
 * The implementation assumes <strong>cross entropy cost function</strong> which makes it straightforward to define update
 * rule as ΔWk(n) = e(n) * η * x(n).
 *
 * The user can chose between three gradient descent flavours:
 * <ul>
 *  <li>stochastic with {@link #stochastic(Map, double[], Vocabulary)}</li>
 *  <li>mini batch with {@link #miniBatch(Map, double[], int, Vocabulary)}</li>
 * </ul>
 * and either of these methods can be used to satisfy {@link TextOptimizer#optimize(Map, double[])} method via functional invocation.
 *
 * <p>
 * For example: <strong>{@literal TextOptimizer optimizer = (x, w) -> new SparseTextGradientDescent(0.0003, 40_000, 200).stochastic(x, w, v)}</strong>.
 * </p>
 *
 * Text classification assumes very large number of features(words) and therefore operations on vectors of such dimensions are
 * infeasible on classic CPU. Given sparse text, only coefficients for words present in the text should be modified. There is a
 * {@link Vocabulary} which lets the algorithm inspect only relevant coefficients for the given words.
 *
 * @author dtemraz
 */
@RequiredArgsConstructor
public class SparseTextGradientDescent {

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final SquaredErrorStoppingCriteria stoppingCriteria; // early termination based on the sum of squared epoch error components
    private final double lambda; // regularization constant for l2 regularization
    private final boolean verbose; // prints epoch and end epoch error at the end of each iteration

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if up to <em>patience</em> number of epochs better solution was not found.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria early termination criteria for the sum of squared epoch error components
     */
    public SparseTextGradientDescent(double learningRate, int epochs, SquaredErrorStoppingCriteria stoppingCriteria) {
        this(learningRate, epochs, stoppingCriteria, 0, false);
    }

    /**
     * Optimizes <em>coefficients</em> for classification of <em>data</em> with stochastic gradient descent.
     *
     * @param data for which to learn regression coefficients
     * @param coefficients for classification of data in logistic regression
     * @param vocabulary of words and their indexes
     */
    public void stochastic(Map<Double, List<String[]>> data, double[] coefficients, Vocabulary vocabulary) {
        sgdCoefficients(TextSample.extractSamples(data, vocabulary), coefficients, vocabulary);
    }

    // performs stochastic optimization of coefficients
    private void sgdCoefficients(TextSample[] trainingSet, double[] coefficients, Vocabulary vocabulary) {
        int samples = trainingSet.length;
        int epoch;
        int features = vocabulary.size();
        double squaredError = -1; // sum of squared epoch error components, putting it outside for printing purposes

        for (epoch = 0; epoch < epochs; epoch++) {
            Vector.shuffle(trainingSet); // this is required to remove data order bias
            double[] epochError = new double[samples];

            // updates coefficients for each sample in epoch
            for (int sample = 0; sample < samples; sample++) {
                TextSample textSample = trainingSet[sample];
                double error = getError(coefficients, features, textSample);
                epochError[sample] = error;
                updateCoefficients(textSample.terms, coefficients, error * learningRate);
            }

            // keep track of lowest found error, exiting if there is no progress for patience epochs
            squaredError = Vector.squaredSum(epochError);
            if (verbose) {
                System.out.println("epoch: " + epoch + " , squared error: " + squaredError);
            }
            if (stoppingCriteria.test(squaredError)) {
                break;
            }
        }
        System.out.println(String.format("converged in: %d epochs, epoch error: %.6f", epoch, squaredError));
    }

    /**
     * Optimizes <em>coefficients</em> for classification of <em>data</em> with mini batch gradient descent.
     *
     * @param data for which to learn regression coefficients
     * @param coefficients for classification of data in logistic regression
     * @param vocabulary of words and their indexes
     * @param batchSize number of samples in epoch to process before batch update is made
     */
    public void miniBatch(Map<Double, List<String[]>> data, double[] coefficients, int batchSize, Vocabulary vocabulary) {
        batchCoefficients(TextSample.extractSamples(data, vocabulary), coefficients, batchSize, vocabulary);
    }

    // performs batch optimization of theta, where batch size <= sample.size
    private void batchCoefficients(TextSample[] trainingSet, double[] coefficients, int batchSize, Vocabulary vocabulary) {
        int totalSamples = trainingSet.length;
        batchSize = Math.min(batchSize, totalSamples);
        int features = vocabulary.size();
        int epoch;
        double updateFactor = learningRate / batchSize;
        double squaredError = -1; // sum of squared epoch error components, putting it outside for printing purposes

        for (epoch = 0; epoch < epochs; epoch++) {
            Vector.shuffle(trainingSet);
            double[] gradient = new double[features + 1]; // change of weights is proportional to gradient, + 1 feature for bias
            double[] epochError = new double[totalSamples]; // vector of error per sample in this epoch
            int batchCount = 0; // count of samples in this batch

            // calculate gradient for each sample in a batch
            for (int sample = 0; sample < totalSamples; sample++) {
                TextSample textSample = trainingSet[sample];
                double error = getError(coefficients, features, textSample);
                // e(n) * x(n), we will apply learning rate divided by batch size after we have seen all samples in a batch
                epochError[sample] = error;
                // accumulate gradient per weight until there are enough samples for a batch update
                updateGradient(textSample.terms, gradient, error);

                batchCount++;
                if (batchCount == batchSize || sample == totalSamples - 1) {
                    // ΔWk(n) = e(n) * η * x(n), gradient contains e(n) * x(n), apply learning rate on gradient for final value
                    for (int i = 0; i < coefficients.length; i++) {
                        coefficients[i] += (gradient[i] * updateFactor) - (lambda * updateFactor * coefficients[i]);
                    }
                    batchCount = 0;
                }
            }

            // keep track of lowest found error, exiting if there is no progress for patience epochs
            squaredError = Vector.squaredSum(epochError);
            if (verbose) {
                System.out.println("epoch: " + epoch + " , squared error: " + squaredError);
            }
            if (stoppingCriteria.test(squaredError)) {
                break;
            }
        }
        System.out.println(String.format("converged in: %d epochs, epoch error: %.6f", epoch, squaredError));
    }


    // calculates error between predicted and expected class
    private static double getError(double[] coefficients, int bias, TextSample textSample) {
        // sigmoid (bias + input*coefficients)
        double estimate = Activation.SIGMOID.apply(coefficients[bias] + dotProduct(textSample.terms, coefficients));
        // difference as expected class minus estimated value
        return textSample.classId - estimate;
    }

    // calculates sums of words tf-idf and theta coefficients
    private static double dotProduct(TF_IDF_Term[] terms, double[] theta) {
        double sum = 0;
        for (TF_IDF_Term term : terms) {
            sum += term.getTfIdf() * theta[term.getId()];
        }
        return sum;
    }

    // updates gradient for this sample proportional to TF-IDF and update value
    private static void updateGradient(TF_IDF_Term[] terms, double[] coefficients, double update) {
        for (TF_IDF_Term term : terms) {
            coefficients[term.getId()] += term.getTfIdf() * update;
        }
        int bias = coefficients.length - 1; // bias coefficient which is in the last position
        coefficients[bias] += update; // bias has value always ON, or in practice 1
    }

    // updates coefficients and bias with value proportional to TF-IDF and update value applying L2 regularization if lambda > 0
    private void updateCoefficients(TF_IDF_Term[] terms, double[] coefficients, double update) {
        L2Regularization.update(terms, coefficients, update, lambda, learningRate);
        int bias = coefficients.length - 1; // bias coefficient which is in the last position
        coefficients[bias] += update; // bias has value always ON, or in practice 1
    }

}
