package algorithms.linear_regression;

import algorithms.linear_regression.optimization.text.MultiClassTextOptimizer;
import algorithms.model.ClassificationResult;
import algorithms.model.TextModel;
import algorithms.model.TextModelWithProbability;
import algorithms.neural_net.StableSoftMaxActivation;
import lombok.extern.log4j.Log4j2;
import structures.text.TF_IDF_Term;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * This class implements SoftMax regression which lets users classify into multiple classes. SoftMax classifier is <strong>discriminative</strong> classifier,
 * and as such tends to give better results when there is exactly one class that matches user input.
 * <p>
 * There is a factory method {@link #getTextModel(Vocabulary, Map, MultiClassTextOptimizer)} which initializes and trains SoftMaxRegression for textual classification.
 * The user is able to classify text via {@link #classify(String[])} into one of the classes defined in training set.
 * </p>
 *
 * @author dtemraz
 */
@Log4j2
public class SoftMaxRegression implements TextModelWithProbability {

    private final Vocabulary vocabulary; // indexed words and their IDF values
    private final Map<Double, double[]> theta = new HashMap<>(); // coefficients for each class
    private final Map<Double, Double> bias = new HashMap<>(); // bias terms for each class
    private final Double[] classes; // cache classes so there is no need to create array for each input

    /**
     * Returns {@link TextModelWithProbability} instance trained with <em>softMaxOptimizer</em> over <em>trainingSet</em>
     * which can be used for multi class classification of textual data.
     * <p>
     * This model will <em>additionally</em> offer method to return probabilities of each class individual class, resulting vector
     * will maintain ordering defined in trainingSet#keySet.
     * </p>
     *
     * @param vocabulary       which defines possible words, their global indexes and IDF values
     * @param trainingSet      map of classes and messages broken into words
     * @param softMaxOptimizer instance to train classifier for textual classification
     * @return {@link TextModel} instance which can be used to classify text into multiple classes
     */
    public static TextModelWithProbability getTextModelWithProbabilities(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassTextOptimizer softMaxOptimizer) {
        return new SoftMaxRegression(vocabulary, trainingSet, softMaxOptimizer);
    }

    /**
     * Returns {@link TextModel} instance trained with <em>softMaxOptimizer</em> over <em>trainingSet</em> which can be used for multi class
     * classification of textual data.
     *
     * @param vocabulary       which defines possible words, their global indexes and IDF values
     * @param trainingSet      map of classes and messages broken into words
     * @param softMaxOptimizer instance to train classifier for textual classification
     * @return {@link TextModel} instance which can be used to classify text into multiple classes
     */
    public static TextModel getTextModel(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassTextOptimizer softMaxOptimizer) {
        return new SoftMaxRegression(vocabulary, trainingSet, softMaxOptimizer);
    }

    // initializes and trains instance of SoftMax classifier for text classification
    private SoftMaxRegression(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassTextOptimizer softMaxOptimizer) {
        this.vocabulary = vocabulary;
        Map<Double, double[]> classCoefficients = initializeTrainingCoefficients(vocabulary, trainingSet);
        train(trainingSet, classCoefficients, softMaxOptimizer);
        updateCoefficients(classCoefficients);
        classes = theta.keySet().toArray(new Double[theta.size()]);
    }

    @Override
    public double classify(String[] words) {
        double[] activations = activations(words);
        return classes[Vector.maxComponentId(activations)];
    }

    @Override
    public ClassificationResult classifyWithProb(String[] words) {
        // find probability of each class and chose one with max probability score
        double[] activations = activations(words);
        int maxComponentId = Vector.maxComponentId(activations);

        return new ClassificationResult(classes[maxComponentId], activations[maxComponentId], activations);
    }

    // compute probabilities of each class with sum normalized to 1
    private double[] activations(String[] words) {
        double[] weightedInput = new double[classes.length];
        TF_IDF_Term[] tf_idf_terms = TF_IDF_Vectorizer.tfIdf(words, vocabulary);
        // calculate weighted input X*theta for each class
        for (int classId = 0; classId < classes.length; classId++) {
            double clazz = classes[classId];
            // class specific coefficients
            double[] coefficients = theta.get(clazz);
            double sum = 0;
            for (TF_IDF_Term term : tf_idf_terms) {
                sum += term.getTfIdf() * coefficients[term.getId()];
            }
            // weighted input for each class in classes array, ordering is same between these arrays
            weightedInput[classId] = sum + bias.get(clazz);
        }
        return StableSoftMaxActivation.apply(weightedInput);
    }

    /*
     * methods for model fitting
     */

    // optimizes class coefficients for SoftMax classification
    private void train(Map<Double, List<String[]>> trainingSet, Map<Double, double[]> classCoefficients, MultiClassTextOptimizer softMaxOptimizer) {
        long before = System.currentTimeMillis();
        softMaxOptimizer.optimize(trainingSet, classCoefficients);
        long after = System.currentTimeMillis();
        log.info("training time: " + TimeUnit.MILLISECONDS.toSeconds(after - before));
    }

    // updates state of internal coefficients and bias for each class
    private void updateCoefficients(Map<Double, double[]> classCoefficients) {
        classCoefficients.forEach((id, coefficients) -> {
            theta.put(id, Arrays.copyOfRange(coefficients, 0, coefficients.length - 1));
            bias.put(id, coefficients[coefficients.length - 1]);
        });
    }

    // initializes coefficients for each class in random value between 0 and 1
    private Map<Double, double[]> initializeTrainingCoefficients(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet) {
        Map<Double, double[]> classCoefficients = new HashMap<>();
        trainingSet.forEach((id, samples) -> {
            // coefficients for each feature and + 1 for bias
            classCoefficients.put(id, Vector.randomArray(vocabulary.size() + 1));
        });
        return classCoefficients;
    }

}