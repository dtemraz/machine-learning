package algorithms.linear_regression.optimization.text;

import algorithms.neural_net.SoftMaxActivation;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;
import utilities.Vector;

import java.util.Map;

/**
 * @author dtemraz
 */
public class SoftMaxOptimizer {

    private void train(TextSample[] trainingSet, Map<Double, double[]> coefficients, Vocabulary vocabulary) {
        int samples = trainingSet.length;
        int epoch;
        Vector.shuffle(trainingSet);
        Double[] classes = coefficients.keySet().toArray(new Double[coefficients.size()]);
        for (epoch = 0; epoch < 1000; epoch++) {
            double[] prods = new double[classes.length];
            // updates coefficients for each sample in epoch
            for (int sample = 0; sample < samples; sample++) {
                TextSample textSample = trainingSet[sample];
                for (int i = 0; i < classes.length; i++) {
                    double classId = classes[i];
                    double[] theta = coefficients.get(classId);
                    double bias = theta[theta.length - 1];
                    prods[i] = bias + dotProduct(textSample.terms, theta);
                }

                double[] activations = SoftMaxActivation.softMax(prods);
                for (int i = 0; i < classes.length; i++) {
                    double classId = classes[i];
                    double expected = textSample.classId == classId ? 1 : 0;
                    double error = expected - activations[i];
                    updateCoefficients(textSample.terms, coefficients.get(classId), error * 0.00004);
                }
            }
        }

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
    private static void updateCoefficients(TF_IDF_Term[] terms, double[] coefficients, double update) {
        for (TF_IDF_Term term : terms) {
            coefficients[term.getId()] += term.getTfIdf() * update;
        }
        int bias = coefficients.length - 1; // bias coefficient which is in the last position
        coefficients[bias] += update; // bias has value always ON, or in practice 1
    }

}
