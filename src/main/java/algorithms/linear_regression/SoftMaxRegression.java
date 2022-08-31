package algorithms.linear_regression;

import algorithms.linear_regression.optimization.multiclass.MultiClassOptimizer;
import algorithms.model.ClassificationResult;
import algorithms.model.Model;
import algorithms.model.TextModel;
import algorithms.model.TextModelWithProbability;
import algorithms.neural_net.StableSoftMaxActivation;
import lombok.extern.log4j.Log4j2;
import structures.text.TF_IDF_Term;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Vocabulary;
import textEmbedding.PoolingWithEmbedding;
import utilities.math.Vector;

import java.util.*;
import java.util.function.Function;

/**
 * This class implements SoftMax regression which lets users classify into multiple classes. SoftMax classifier is <strong>discriminative</strong> classifier,
 * and as such tends to give better results when there is exactly one class that matches user input.
 * <p>
 * There is a factory method {@link #getTextModel(Vocabulary, Map, MultiClassOptimizer)} which initializes and trains SoftMaxRegression for textual classification.
 * The user is able to classify text via {@link #classify(String[])} into one of the classes defined in training set.
 * </p>
 *
 * @author dtemraz
 */
@Log4j2
public class SoftMaxRegression implements TextModelWithProbability, Model {

    private final Vocabulary vocabulary; // indexed words and their IDF values
    private final Map<Double, double[]> theta = new HashMap<>(); // coefficients for each class
    private final Map<Double, Double> bias = new HashMap<>(); // bias terms for each class
    private final Double[] classes; // cache classes so there is no need to create array for each input
    private final Function<String[], double[]> activationFunc; // apply softmax on tf_idf or word embeddings
    private final PoolingWithEmbedding poolingWithEmbedding;


    /**
     * Returns {@link TextModelWithProbability} instance trained with <em>softMaxOptimizer</em> over <em>trainingSet</em>
     * which can be used for multi class classification of textual data.
     * <p>
     * This model will <em>additionally</em> offer method to return probabilities of each class individual class, resulting vector
     * will maintain ordering defined in trainingSet#keySet.
     * </p>
     *
     * @param vocabulary which defines possible words, their global indexes and IDF values
     * @param trainingSet map of classes and messages broken into words
     * @param softMaxOptimizer instance to train classifier for textual classification
     * @return {@link TextModel} instance which can be used to classify text into multiple classes
     */
    public static TextModelWithProbability getTextModelWithProbabilities(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassOptimizer softMaxOptimizer) {
        return new SoftMaxRegression(vocabulary, trainingSet, softMaxOptimizer);
    }

    /**
     * Returns {@link TextModel} instance trained with <em>softMaxOptimizer</em> over <em>trainingSet</em> which can be used for multi class
     * classification of textual data.
     *
     * @param vocabulary which defines possible words, their global indexes and IDF values
     * @param trainingSet map of classes and messages broken into words
     * @param softMaxOptimizer instance to train classifier for textual classification
     * @return {@link TextModel} instance which can be used to classify text into multiple classes
     */
    public static TextModel getTextModel(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassOptimizer softMaxOptimizer) {
        return new SoftMaxRegression(vocabulary, trainingSet, softMaxOptimizer);
    }

    public static TextModel getWordEmbeddingsModel(Map<Double, List<String[]>> trainingSet, PoolingWithEmbedding poolingWithEmbedding, MultiClassOptimizer multiClassOptimizer) {
        return new SoftMaxRegression(trainingSet, poolingWithEmbedding, multiClassOptimizer);
    }

    // initializes and trains instance of SoftMax classifier for text classification
    private SoftMaxRegression(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, MultiClassOptimizer softMaxOptimizer) {
        this.vocabulary = vocabulary;
        Map<Double, double[]> classCoefficients = initializeTrainingCoefficients(vocabulary.size(), trainingSet.keySet());
        softMaxOptimizer.optimize(trainingSet, classCoefficients, vocabulary);
        updateCoefficients(classCoefficients);
        classes = theta.keySet().toArray(Double[]::new);
        this.activationFunc = this::activations;
        poolingWithEmbedding = null;
    }

    private SoftMaxRegression(Map<Double, List<String[]>> trainingSet, PoolingWithEmbedding poolingWithEmbedding, MultiClassOptimizer softMaxOptimizer) {
        this.poolingWithEmbedding = poolingWithEmbedding;
        this.activationFunc = this::wordVectorActivations;
        Map<Double, double[]> classCoefficients = initializeTrainingCoefficients(poolingWithEmbedding.getDimensions(), trainingSet.keySet());
        Map<Double, List<double[]>> embeddedSamples = poolingWithEmbedding.transform(trainingSet);
        softMaxOptimizer.optimize(embeddedSamples, classCoefficients);
        updateCoefficients(classCoefficients);
        classes = theta.keySet().toArray(Double[]::new);
        this.vocabulary = null;
    }

    @Override
    public double classify(String[] words) {
        double[] activations = activationFunc.apply(words);
        return classes[Vector.maxComponentId(activations)];
    }

    @Override
    public ClassificationResult classifyWithProb(String[] words) {
        // find probability of each class and chose one with max probability score
        double[] activations = activationFunc.apply(words);
        int maxComponentId = Vector.maxComponentId(activations);
        return new ClassificationResult(classes[maxComponentId], activations[maxComponentId], activations);
    }

    @Override
    public double predict(double[] features) {
        double[] activations = activations(features);
        return classes[Vector.maxComponentId(activations)];
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

    private double[] wordVectorActivations(String[] words) {
        double[] averagedFeatures = poolingWithEmbedding.transform(words);
        return activations(averagedFeatures);
    }

    private double[] activations(double[] features) {
        // calculate weighted input X*theta for each class
        double[] weightedInput = new double[classes.length];
        for (int classId = 0; classId < classes.length; classId++) {
            double clazz = classes[classId];
            double[] coefficients = theta.get(clazz);
            double sum = 0;
            for (int i = 0; i < features.length; i++) {
                sum += features[i] * coefficients[i];
            }
            // weighted input for each class in classes array, ordering is same between these arrays
            weightedInput[classId] = sum + bias.get(clazz);
        }
        return StableSoftMaxActivation.apply(weightedInput);
    }

    /*
     * methods for model fitting
     */

    // updates state of internal coefficients and bias for each class
    private void updateCoefficients(Map<Double, double[]> classCoefficients) {
        classCoefficients.forEach((id, coefficients) -> {
            theta.put(id, Arrays.copyOfRange(coefficients, 0, coefficients.length - 1));
            bias.put(id, coefficients[coefficients.length - 1]);
        });
    }

    // initializes coefficients for each class in random value between 0 and 1
    private Map<Double, double[]> initializeTrainingCoefficients(int size, Set<Double> classes) {
        Map<Double, double[]> classCoefficients = new HashMap<>();
        // coefficients for each class/feature and + 1 for bias at the end
        classes.forEach(c -> classCoefficients.put(c, Vector.randomArray(size + 1)));
        return classCoefficients;
    }

}
