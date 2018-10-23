package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.StableSoftMaxActivation;
import com.google.common.util.concurrent.AtomicDoubleArray;
import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * This class implements <strong>parallel</strong>stochastic gradient descent(hogwild) for SoftMax regression. It has the same properties
 * as {@link SoftMaxOptimizer} except it runs in parallel.
 * <p>
 * The class should offer performance to sequential variant which directly scales with number of cores.
 * </p>
 */
@RequiredArgsConstructor
public class ParallelSoftMaxOptimizer {

    private static final double TARGET = 1; // there is only one true class corresponding to a sample
    private static final double OTHER = 0; // all other classes are false and optimizer should be trained to converge activation to 0 for these classes
    private static final double SHUFFLE_THRESHOLD = 0.1;

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final double lambda; // regularization constant for l2 regularization
    private final boolean verbose; // prints epoch and end epoch error at the end of each iteration

    // by default do not log epoch errors
    public ParallelSoftMaxOptimizer(double learningRate, int epochs) {
        this(learningRate, epochs, 0, false);
    }

    public ParallelSoftMaxOptimizer(double learningRate, double l2lambda, int epochs) {
        this(learningRate, epochs, l2lambda, false);
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
    public void stochastic(Map<Double, List<String[]>> data, Map<Double, double[]> coefficients, Vocabulary vocabulary) {
        train(TextSample.extractSamples(data, vocabulary), coefficients);
    }

    private void train(TextSample[] trainingSet, Map<Double, double[]> coefficients) {
        Vector.shuffle(trainingSet);
        // concurrent coefficients alias for multi threaded update
        Map<Double, AtomicDoubleArray> concurrentCoefficients = toConcurrent(coefficients);
        // maintain class iteration order for all methods, save multiple needless array conversions
        Double[] classes = concurrentCoefficients.keySet().toArray(new Double[coefficients.size()]);
        // accumulate epoch error for each class
        Map<Double, DoubleAdder> errorAccumulators = initializeErrorAdders(classes);
        int epoch;
        for (epoch = 0; epoch < epochs; epoch++) {
            if (Math.random() <= SHUFFLE_THRESHOLD) {
                Vector.shuffle(trainingSet);
            }
            // updates coefficients for each sample in epoch
            Arrays.stream(trainingSet).parallel().forEach(txt -> {
                // calculate weighted input(X * W) for each class
                double[] weightedInput = calculateWeightedInput(txt, classes, concurrentCoefficients);
                // calculate SoftMax activation for each class
                double[] activations = StableSoftMaxActivation.apply(weightedInput);
                // update weights for each class given its activation and error for the sample
                for (int i = 0; i < classes.length; i++) {
                    double classId = classes[i];
                    // only one class can be true class for a given sample, for all other classes value should be zero
                    double expected = txt.classId == classId ? TARGET : OTHER;
                    // weights of target class should converge to 1, for all other classes weights should converge to 0
                    double error = expected - activations[i];
                    updateCoefficients(txt.terms, concurrentCoefficients.get(classId), learningRate * error);
                    if (verbose) {
                        errorAccumulators.get(classId).add(error * error);
                    }
                }
            });
            if (verbose) {
                printAverageEpochError(epoch, errorAccumulators);
            }
        }

        writeBack(concurrentCoefficients, coefficients);
    }

    // prepare initial error accumulators for each class
    private static Map<Double, DoubleAdder> initializeErrorAdders(Double[] classes) {
        Map<Double, DoubleAdder> classErrorAdders = new ConcurrentHashMap<>();
        for (Double c : classes) {
            classErrorAdders.put(c, new DoubleAdder());
        }
        return classErrorAdders;
    }

    private static double[] calculateWeightedInput(TextSample textSample, Double[] classes, Map<Double, AtomicDoubleArray> coefficients) {
        double[] weightedInput = new double[classes.length];
        for (int i = 0; i < classes.length; i++) {
            AtomicDoubleArray classCoefficients = coefficients.get(classes[i]);
            double bias = classCoefficients.get(classCoefficients.length() - 1);
            weightedInput[i] = bias + dotProduct(textSample.terms, classCoefficients);
        }
        return weightedInput;
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
        // bias coefficient which is in the last position, should not be regularized
        int bias = coefficients.length() - 1;
        coefficients.addAndGet(bias, update); // bias has value always ON, or in practice 1
    }

    // creates concurrent copy of the coefficients which supports lock free update
    private static Map<Double, AtomicDoubleArray> toConcurrent(Map<Double, double[]> coefficients) {
        Map<Double, AtomicDoubleArray> concurrentCoefficients = new ConcurrentHashMap<>();
        for (Map.Entry<Double, double[]> entry : coefficients.entrySet()) {
            double[] theta = entry.getValue();
            AtomicDoubleArray concurrent = new AtomicDoubleArray(theta.length);
            for (int i = 0; i < theta.length; i++) {
                concurrent.set(i, theta[i]);
            }
            concurrentCoefficients.put(entry.getKey(), concurrent);
        }
        return concurrentCoefficients;
    }

    // copies calculated concurrent coefficients into caller's coefficients
    private static void writeBack(Map<Double, AtomicDoubleArray> concurrentCoefficients, Map<Double, double[]> coefficients) {
        for (Map.Entry<Double, AtomicDoubleArray> entry : concurrentCoefficients.entrySet()) {
            double[] theta = coefficients.get(entry.getKey());
            AtomicDoubleArray concurrent = entry.getValue();
            for (int i = 0; i < concurrent.length(); i++) {
                theta[i] = concurrent.get(i);
            }
        }
    }

    // prints current epoch and average epoch error across all classes
    private static void printAverageEpochError(int epoch, Map<Double, DoubleAdder> errorAccumulators) {
        double totalError = 0;
        for(DoubleAdder accumulator : errorAccumulators.values()) {
            totalError += accumulator.sumThenReset();
        }
        double averageError = totalError / errorAccumulators.size();
        System.out.println("epoch: " + epoch + " , average error: " + averageError);
    }

}
