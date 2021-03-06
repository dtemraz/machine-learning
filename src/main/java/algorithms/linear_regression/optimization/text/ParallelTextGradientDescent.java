package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.Activation;
import com.google.common.util.concurrent.AtomicDoubleArray;
import lombok.extern.log4j.Log4j2;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * This class implements parallel gradient descent optimization for sparse text classification with logistic regression.
 * The implementation assumes <strong>cross entropy cost function</strong> which makes it straightforward to define update
 * rule as ΔWk(n) = e(n) * η * x(n).
 *
 * <p>
 * The only supported method currently is parallel <strong>stochastic</strong> gradient descent, hogwild specification.*
 * </p>
 *
 * The algorithm is pretty good for <strong>sparse</strong> vectors(short text analysis) since each thread will inspect very tiny subset
 * of features concurrently. Due to sparsity, contention and mixed state updates should be reduced to very rare events.
 *
 * The algorithm should offer measurable performance lift when there are more than 10_000 samples and in those scenarios
 * should be preferred to {@link TextGradientDescent#stochastic(Map, double[], Vocabulary)} optimization.
 *
 * @see <a href="https://arxiv.org/pdf/1106.5730.pdf">hogwild</a>
 *
 * @author dtemraz
 */
@Log4j2
public class ParallelTextGradientDescent {

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final SquaredErrorStoppingCriteria stoppingCriteria; // early termination based on the sum of squared epoch error components
    private final double lambda; // regularization constant for l2 regularization
    private final boolean verbose; // prints epoch error at the end of each epoch if set to true

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if up to <em>patience</em> number of epochs better solution was not found.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria early termination criteria for the sum of squared epoch error components
     */
    public ParallelTextGradientDescent(double learningRate, int epochs, SquaredErrorStoppingCriteria stoppingCriteria) {
        this(learningRate, epochs, stoppingCriteria, 0, false);
    }

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if up to <em>patience</em> number of epochs better solution was not found.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria early termination criteria for the sum of squared epoch error components
     * @param lambda regularization penalty
     * @param verbose logs epoch number and squared error in epoch
     */
    public ParallelTextGradientDescent(double learningRate, int epochs, SquaredErrorStoppingCriteria stoppingCriteria, double lambda, boolean verbose) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.stoppingCriteria = stoppingCriteria;
        this.lambda = lambda;
        this.verbose = verbose;
    }

    /**
     * Optimizes <em>coefficients</em> for classification of <em>data</em> with parallel stochastic gradient descent.
     *
     * <p>The algorithm should offer performance lift compared to sequential mode when there are more than 10000 samples.</p>
     *
     * @param data for which to learn regression coefficients
     * @param coefficients for classification of data in logistic regression
     * @param vocabulary of words and their indexes
     */
    public void stochastic(Map<Double, List<String[]>> data, double[] coefficients, Vocabulary vocabulary) {
        sgdCoefficients(TextSample.extractSamples(data, vocabulary), coefficients);
    }

    // performs stochastic optimization of coefficients
    private void sgdCoefficients(TextSample[] trainingSet, double[] coefficients) {
        AtomicDoubleArray concurrentCoefficients = toConcurrent(coefficients);
        // accumulator is better choice than atomic long since we need epoch error only once and after all threads have finished
        DoubleAdder errorAccumulator = new DoubleAdder();
        double squaredError = -1; // sum of squared epoch error components, putting it outside for printing purposes
        int epoch;
        for (epoch = 0; epoch < epochs; epoch++) {
            Vector.shuffle(trainingSet);
            // updates coefficients in parallel for each sample in epoch
            Arrays.stream(trainingSet).parallel().forEach(txt -> {
                double error = getError(concurrentCoefficients, txt);
                updateCoefficients(txt.terms, concurrentCoefficients, error * learningRate);
                errorAccumulator.add(error * error);
            });
            // now that we went through all the samples, we can reduce epoch error to sum of squares
            squaredError = errorAccumulator.sumThenReset() / trainingSet.length;
            if (verbose) {
                log.info("epoch: " + epoch + " , squared error: " + squaredError);
            }
            if (stoppingCriteria.test(squaredError)) {
                break;
            }
        }
        // copy calculated coefficients into original coefficients
        writeBack(concurrentCoefficients, coefficients);
        log.info(String.format("converged in: %d epochs, epoch error: %.6f", epoch, squaredError));
    }

    // calculates error between predicted and expected class
    private static double getError(AtomicDoubleArray coefficients, TextSample textSample) {
        double bias = coefficients.get(coefficients.length() - 1);
        double estimate = Activation.SIGMOID.apply(bias + dotProduct(textSample.terms, coefficients));
        // difference as expected class minus estimated value
        return textSample.classId - estimate;
    }

    // calculates sums of words tf-idf and theta coefficients
    private static double dotProduct(TF_IDF_Term[] terms, AtomicDoubleArray theta) {
        double sum = 0;
        for (TF_IDF_Term term : terms) {
            sum += term.getTfIdf() * theta.get(term.getId());
        }
        return sum;
    }

    // updates coefficients and bias with value proportional to TF-IDF and update value
    private void updateCoefficients(TF_IDF_Term[] terms, AtomicDoubleArray coefficients, double update) {
        L2Regularization.update(terms, coefficients, update, lambda, learningRate);
        int bias = coefficients.length() - 1; // bias coefficient which is in the last position
        coefficients.addAndGet(bias, update); // bias has value always ON, or in practice 1
    }

    // creates concurrent copy of the coefficients which supports lock free update
    private static AtomicDoubleArray toConcurrent(double[] coefficients) {
        AtomicDoubleArray concurrent = new AtomicDoubleArray(coefficients.length);
        for (int i = 0; i < coefficients.length; i++) {
            concurrent.set(i, coefficients[i]);
        }
        return concurrent;
    }

    // copies calculated concurrent coefficients into caller's coefficients
    private static void writeBack(AtomicDoubleArray concurrentCoefficients, double[] coefficients) {
        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = concurrentCoefficients.get(i);
        }
    }

}