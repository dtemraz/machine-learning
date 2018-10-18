package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.Activation;
import com.google.common.util.concurrent.AtomicDouble;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.Vector;

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
 * should be preferred to {@link algorithms.linear_regression.optimization.text.SparseTextGradientDescent#stochastic(Map, double[], Vocabulary)} optimization.
 *
 * @see <a href="https://arxiv.org/pdf/1106.5730.pdf">hogwild</a>
 *
 * @author dtemraz
 */
public class ParallelSparseTextGradientDescent {

    private static final double SHUFFLE_THRESHOLD = 0.05;

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final SquaredErrorStoppingCriteria stoppingCriteria; // early termination based on the sum of squared epoch error components
    private final boolean verbose;

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if up to <em>patience</em> number of epochs better solution was not found.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria early termination criteria for the sum of squared epoch error components
     */
    public ParallelSparseTextGradientDescent(double learningRate, int epochs, SquaredErrorStoppingCriteria stoppingCriteria) {
        this(learningRate, epochs, stoppingCriteria, false);
    }

    /**
     * Constructs instance of gradient descent with <em>learningRate</em> which will run for most <em>epochs</em>. This
     * instance will stopping learning if up to <em>patience</em> number of epochs better solution was not found.
     *
     * @param learningRate by which gradient descent optimize theta, smaller = stable, larger = faster
     * @param epochs number of epochs algorithm will run at most
     * @param stoppingCriteria early termination criteria for the sum of squared epoch error components
     * @param verbose logs epoch number and squared error in epoch
     */
    public ParallelSparseTextGradientDescent(double learningRate, int epochs, SquaredErrorStoppingCriteria stoppingCriteria, boolean verbose) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.stoppingCriteria = stoppingCriteria;
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
        sgdCoefficients(TextSample.extractSamples(data, vocabulary), coefficients, vocabulary);
    }

    // performs stochastic optimization of coefficients
    private void sgdCoefficients(TextSample[] trainingSet, double[] coefficients, Vocabulary vocabulary) {
        int epoch;
        int features = vocabulary.size(); // bias weight comes after all attribute weights
        AtomicDouble[] concurrentCoefficients = toConcurrent(coefficients);
        // accumulator is better choice than atomic long since we need epoch error only once and after all threads have finished
        DoubleAdder errorAccumulator = new DoubleAdder();
        double squaredError = -1; // sum of squared epoch error components, putting it outside for printing purposes
        Vector.shuffle(trainingSet);

        for (epoch = 0; epoch < epochs; epoch++) {
            // shuffling may speed up convergence, however shuffling is a bottleneck since this is sequential implementation, therefore
            // allow small probability that it randomly occurs. This is okay since (parallel) SGD is already fuzzy.
            // it's not trivial to implement parallel shuffle which beats sequential one and is uniform at the same time
            if (Math.random() <= SHUFFLE_THRESHOLD) {
                Vector.shuffle(trainingSet);
            }
            // updates coefficients in parallel for each sample in epoch
            Arrays.stream(trainingSet).parallel().forEach(txt -> {
                double error = getError(concurrentCoefficients, features, txt);
                updateCoefficients(txt.terms, concurrentCoefficients, error * learningRate);
                errorAccumulator.add(error * error);
            });
            // now that we went through all the samples, we can reduce epoch error to sum of squares
            squaredError = errorAccumulator.sumThenReset();
            if (verbose) {
                System.out.println("epoch: " + epoch + " , squared error: " + squaredError);
            }
            if (stoppingCriteria.test(squaredError)) {
                break;
            }
        }
        // copy calculated coefficients into original coefficients
        writeBack(concurrentCoefficients, coefficients);
        System.out.println(String.format("converged in: %d epochs, epoch error: %.6f", epoch, squaredError));
    }

    // calculates error between predicted and expected class
    private static double getError(AtomicDouble[] coefficients, int biasIndex, TextSample textSample) {
        double bias = coefficients[biasIndex].get();
        double estimate = Activation.SIGMOID.apply(bias + dotProduct(textSample.terms, coefficients));
        // difference as expected class minus estimated value
        return textSample.classId - estimate;
    }

    // calculates sums of words tf-idf and theta coefficients
    private static double dotProduct(TF_IDF_Term[] terms, AtomicDouble[] theta) {
        double sum = 0;
        for (TF_IDF_Term term : terms) {
            sum += term.getTfIdf() * theta[term.getId()].get();
        }
        return sum;
    }

    // updates coefficients and bias with value proportional to TF-IDF and update value
    private static void updateCoefficients(TF_IDF_Term[] terms, AtomicDouble[] coefficients, double update) {
        for (TF_IDF_Term term : terms) {
            coefficients[term.getId()].addAndGet(term.getTfIdf() * update);
        }
        int bias = coefficients.length - 1; // bias coefficient which is in the last position
        coefficients[bias].addAndGet(update); // bias has value always ON, or in practice 1
    }

    // creates concurrent copy of the coefficients which supports lock free update
    private static AtomicDouble[] toConcurrent(double[] coefficients) {
        AtomicDouble[] concurrent = new AtomicDouble[coefficients.length];
        for (int i = 0; i < coefficients.length; i++) {
            concurrent[i] = new AtomicDouble(coefficients[i]);
        }
        return concurrent;
    }

    // copies calculated concurrent coefficients into caller's coefficients
    private static void writeBack(AtomicDouble[] concurrentCoefficients, double[] coefficients) {
        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = concurrentCoefficients[i].get();
        }
    }

}