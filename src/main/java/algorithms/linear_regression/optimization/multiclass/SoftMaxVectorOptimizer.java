package algorithms.linear_regression.optimization.multiclass;

import algorithms.linear_regression.optimization.text.L2Regularization;
import algorithms.neural_net.StableSoftMaxActivation;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import structures.Sample;
import utilities.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This class implements stochastic gradient descent optimization for dense vector classification with SoftMax regression.
 * The implementation assumes <strong>cross entropy cost function</strong> which makes it straightforward to define update
 * rule as ΔWk(n) = e(n) * η * x(n).
 * <p>
 * Additionally, user is able to specify L2 regularization penalty <em>lambda</em> which changes update rule into: ΔWk(n) = (e(n) * η * x(n)) - lambda*Wk
 * </p>
 *
 * @author dtemraz
 */
@RequiredArgsConstructor
@Log4j2
class SoftMaxVectorOptimizer {

    private static final double TARGET = 1; // there is only one true class corresponding to a sample
    private static final double OTHER = 0; // all other classes are false and optimizer should be trained to converge activation to 0 for these classes


    private final double learningRate; // proportion of gradient by which we take next step
    private final int epochs; // maximal number of epochs the algorithm will run
    private final double lambda; // regularization constant for l2 regularization
    private final boolean verbose; // prints epoch and end epoch error at the end of each iteration


    // by default do not log epoch errors
    public SoftMaxVectorOptimizer(double learningRate, int epochs) {
        this(learningRate, epochs, 0, true);
    }

    public SoftMaxVectorOptimizer(double learningRate, double l2lambda, int epochs) {
        this(learningRate, epochs, l2lambda, true);
    }

    public void optimize(Map<Double, List<double[]>> data, Map<Double, double[]> coefficients) {
        stochastic(data, coefficients);
    }

    public void stochastic(Map<Double, List<double[]>> data, Map<Double, double[]> coefficients) {
        Sample[] trainingSet = toSamples(data);
        Double[] classes = coefficients.keySet().toArray(Double[]::new);
        int epoch;
        for (epoch = 0; epoch < epochs; epoch++) {
            Vector.shuffle(trainingSet);
            double[][] epochError = new double[classes.length][trainingSet.length];
            for (int sample = 0; sample < trainingSet.length; sample++) {
                Sample textSample = trainingSet[sample];
                double[] weightedInput = calculateWeightedInput(textSample, classes, coefficients);
                double[] activations = StableSoftMaxActivation.apply(weightedInput);
                for (int classId = 0; classId < classes.length; classId++) {
                    double classLabel = classes[classId];
                    double expected = textSample.getTarget() == classLabel ? TARGET : OTHER;
                    double error = expected - activations[classId];
                    epochError[classId][sample] = error;
                    updateCoefficients(textSample.getValues(), coefficients.get(classLabel), error * learningRate);
                }
            }
            if (verbose) {
                SoftMaxOptimizer.printAverageEpochError(epoch, trainingSet.length, epochError);
            }
        }
    }

    private static double[] calculateWeightedInput(Sample sample, Double[] classes, Map<Double, double[]> coefficients) {
        double[] weightedInput = new double[classes.length];
        for (int i = 0; i < classes.length; i++) {
            double[] classCoefficients = coefficients.get(classes[i]);
            double bias = classCoefficients[classCoefficients.length - 1];
            weightedInput[i] = bias + dotProduct(sample.getValues(), classCoefficients);
        }
        return weightedInput;
    }

    private void updateCoefficients(double[] features, double[] coefficients, double update) {
        // default behaviour is to not apply regularization, lambda = 0
        L2Regularization.update(features, coefficients, update, lambda, learningRate);
        // bias coefficient, which is in the last position, should not be regularized
        int bias = coefficients.length - 1;
        coefficients[bias] += update; // bias has value always ON, or in practice 1
    }

    private Sample[] toSamples(Map<Double, List<double[]>> data) {
        int totalSamples = data.values().stream().map(List::size).reduce(0, Integer::sum);
        List<Sample> samples = new ArrayList<>(totalSamples);
        for (Map.Entry<Double, List<double[]>> classSamples : data.entrySet()) {
            Double classId = classSamples.getKey();
            for (double[] features : classSamples.getValue()) {
                samples.add(new Sample(features, classId));
            }
        }
        return samples.toArray(Sample[]::new);
    }

    private static double dotProduct(double[] features, double[] coefficients) {
        double sum = 0;
        for (int i = 0; i < features.length; i++) {
            // coefficients have +1 length for bias term, which is ignored in this calculation
            sum += features[i] * coefficients[i];
        }
        return sum;
    }

}
