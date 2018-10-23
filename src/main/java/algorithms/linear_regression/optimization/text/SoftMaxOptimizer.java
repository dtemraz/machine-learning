package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.StableSoftMaxActivation;
import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.List;
import java.util.Map;

/**
 * This class implements stochastic gradient descent optimization for sparse text classification with SoftMax regression.
 * The implementation assumes <strong>cross entropy cost function</strong> which makes it straightforward to define update
 * rule as ΔWk(n) = e(n) * η * x(n).
 * <p>
 * Additionally, user is able to specify L2 regularization penalty <em>lambda</em> which changes update rule into: ΔWk(n) = (e(n) * η * x(n)) - lambda*Wk
 * </p>
 *
 * @author dtemraz
 */
@RequiredArgsConstructor
public class SoftMaxOptimizer {

    private static final double TARGET = 1; // there is only one true class corresponding to a sample
    private static final double OTHER = 0; // all other classes are false and optimizer should be trained to converge activation to 0 for these classes

    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final double lambda; // regularization constant for l2 regularization
    private final boolean verbose; // prints epoch and end epoch error at the end of each iteration

    // by default do not log epoch errors
    public SoftMaxOptimizer(double learningRate, int epochs) {
        this(learningRate, epochs, 0, false);
    }

    public SoftMaxOptimizer(double learningRate, double l2lambda, int epochs) {
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
        int epoch;
        Double[] classes = coefficients.keySet().toArray(new Double[coefficients.size()]);
        for (epoch = 0; epoch < epochs; epoch++) {
            double[][] epochError = new double[classes.length][trainingSet.length];
            // updates coefficients for each sample in epoch
            for (int sample = 0; sample < trainingSet.length; sample++) {
                TextSample textSample = trainingSet[sample];
                // calculate weighted input(X * W) for each class
                double[] weightedInput = calculateWeightedInput(textSample, classes, coefficients);
                // calculate SoftMax activation for each class
                double[] activations = StableSoftMaxActivation.apply(weightedInput);
                // update weights for each class given its activation and error for the sample
                for (int classId = 0; classId < classes.length; classId++) {
                    double classLabel = classes[classId];
                    // only one class can be true class for a given sample, for all other classes value should be zero
                    double expected = textSample.classId == classLabel ? TARGET : OTHER;
                    // weights of target class should converge to 1, for all other classes weights should converge to 0
                    double error = expected - activations[classId];
                    epochError[classId][sample] = error;
                    updateCoefficients(textSample.terms, coefficients.get(classLabel), error * learningRate);
                }
            }

            if (verbose) {
                printAverageEpochError(epoch, epochError);
            }
        }
    }

    // calculates input multiplied by weights(dot product) for each class
    private static double[] calculateWeightedInput(TextSample textSample, Double[] classes, Map<Double, double[]> coefficients) {
        double[] weightedInput = new double[classes.length];
        for (int i = 0; i < classes.length; i++) {
            double[] classCoefficients = coefficients.get(classes[i]);
            double bias = classCoefficients[classCoefficients.length - 1];
            weightedInput[i] = bias + dotProduct(textSample.terms, classCoefficients);
        }
        return weightedInput;
    }

    // calculates sums of words tf-idf and theta coefficients
    private static double dotProduct(TF_IDF_Term[] terms, double[] theta) {
        double sum = 0;
        for (TF_IDF_Term term : terms) {
            sum += term.getTfIdf() * theta[term.getId()];
        }
        return sum;
    }

    // updates coefficients and bias with value proportional to TF-IDF and update value
    private void updateCoefficients(TF_IDF_Term[] terms, double[] coefficients, double update) {
        // default behaviour is to not apply regularization, lambda = 0
        L2Regularization.update(terms, coefficients, update, lambda, learningRate);
        // bias coefficient which is in the last position, should not be regularized
        int bias = coefficients.length - 1;
        coefficients[bias] += update; // bias has value always ON, or in practice 1
    }

    // prints current epoch and average epoch error across all classes
    private static void printAverageEpochError(int epoch, double[][] epochError) {
        double totalError = 0;
        int classes = epochError.length;
        // calculate epoch error for each class
        for (double[] error : epochError) {
            totalError += Vector.squaredSum(error);
        }
        double averageError = totalError / classes;
        System.out.println("epoch: " + epoch + " , average error: " + averageError);
    }

}

