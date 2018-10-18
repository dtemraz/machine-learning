package algorithms.linear_regression;

import algorithms.ensemble.model.TextModel;
import algorithms.neural_net.SoftMaxActivation;
import structures.text.TF_IDF_Term;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Vocabulary;

import java.util.List;
import java.util.Map;

/**
 * @author dtemraz
 */
public class SoftMaxClassifier implements TextModel {

    private final Vocabulary vocabulary;
    private final Map<Double, List<String[]>> trainingSet;

    private Map<Double, double[]> theta;
    private Map<Double, Double> bias;

    public SoftMaxClassifier(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet) {
        this.vocabulary = vocabulary;
        this.trainingSet = trainingSet;
    }

    @Override
    public double classify(String[] words) {
        return 0;
    }


    // calculates dot product of words tf-idf and theta coefficients for associated words
    private double maxProbabilityClass(String[] words) {
        Double[] classes = theta.keySet().toArray(new Double[theta.size()]);
        double[] prods = new double[classes.length];
        TF_IDF_Term[] tf_idf_terms = TF_IDF_Vectorizer.tfIdf(words, vocabulary);
        int id = 0;
        for (Double classId : classes) {
            double[] coefficients = theta.get(classId);
            double sum = 0;
            for (TF_IDF_Term term : tf_idf_terms) {
                sum += term.getTfIdf() * coefficients[term.getId()];
            }
            prods[id++] = sum + bias.get(classId);
        }
        return maxProbabilityClass(classes, prods);
    }

    private double maxProbabilityClass(Double[] classes, double[] prods) {
        double[] softMaxOutput = SoftMaxActivation.softMax(prods);
        double max = Double.MAX_VALUE;
        int classIndex = Integer.MIN_VALUE;
        for (int i = 0; i < softMaxOutput.length; i++) {
            double classOutput = softMaxOutput[i];
            if (classOutput > max) {
                max = classOutput;
                classIndex = i;
            }
        }
        return classes[classIndex];
    }

}
