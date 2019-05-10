package algorithms.linear_regression;

import algorithms.linear_regression.optimization.real_vector.Optimizer;
import algorithms.linear_regression.optimization.text.TextOptimizer;
import algorithms.model.Model;
import algorithms.model.TextModel;
import algorithms.neural_net.Activation;
import lombok.extern.log4j.Log4j2;
import structures.text.TF_IDF_Term;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * This class implements logistic regression for {@link TextModel} and {@link Model} types of classification. The interfaces
 * are exposed via factory methods {@link #getTextModel(Vocabulary, Map, TextOptimizer)} and {@link #getModel(List, Optimizer)}
 * respectively to ensure correct initialization of logistic regression instance.
 * <p>
 * The user is able to classify text via {@link #classify(String[])} or non-textual sample via {@link #predict(double[])}.
 * </p>
 *
 * @author dtemraz
 */
@Log4j2
public class LogisticRegression implements TextModel, Model, Serializable {

    private static final long serialVersionUID = 1L;

    private final Vocabulary vocabulary; // index of unique words and their inverse document frequencies

    private final double[] theta; // regression coefficients associated with indexed words
    private final double bias; // intercept term for regression line

    /**
     * Returns {@link TextModel} instance trained with <em>optimizer</em> over <em>trainingSet</em> which can be used for binary
     * classification of textual data.
     *
     * @param vocabulary which defines possible words and their indexes
     * @param trainingSet map of classes and messages broken into words per classes
     * @param optimizer instance to train classifiers with sparse text gradient descent configuration
     * @return TextModel instance which can be used to classify text
     */
    public static TextModel getTextModel(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        return new LogisticRegression(vocabulary, trainingSet, optimizer);
    }

    /**
     * Returns {@link Model} instance trained with <em>optimizer</em> over <em>trainingSet</em> which can be used for binary
     * classification of vectorized data.
     *
     * @param trainingSet list of training samples, <strong>class id</strong>should be last element in the sample array
     * @param optimizer instance to train classifiers with gradient descent configuration
     * @return Model instance which can be used to classify vectorized data
     */
    public static Model getModel(List<double[]> trainingSet, Optimizer optimizer) {
        return new LogisticRegression(trainingSet, optimizer);
    }

    // initializes and trains logistic regression for textual classification of trainingSet
    LogisticRegression(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        this.vocabulary = vocabulary;
        // coefficients for each feature + bias
        double[] coefficients = Vector.randomArray(vocabulary.size() + 1);
        long before = System.currentTimeMillis();
        optimizer.optimize(trainingSet, coefficients);
        long after = System.currentTimeMillis();
        log.info("training time: " + TimeUnit.MILLISECONDS.toSeconds(after - before));
        // this is inconsistent with gradient descent which stores bias in 0th array position - TODO normalize
        bias = coefficients[coefficients.length - 1];
        theta = Arrays.copyOfRange(coefficients, 0, coefficients.length - 1);
    }

    // initializes and trains logistic regression for classification of trainingSet
    LogisticRegression(List<double[]> trainingSet,  Optimizer optimizer) {
        TrainingSet trainingSamples = TrainingSet.build(trainingSet);
        // coefficients for each feature + bias
        double[] coefficients = Vector.randomArray(trainingSet.get(0).length + 1);
        long before = System.currentTimeMillis();
        optimizer.optimize(trainingSamples.input, trainingSamples.expected, coefficients);
        long after = System.currentTimeMillis();
        log.info("training time: " + TimeUnit.MILLISECONDS.toSeconds(after - before));
        // this is inconsistent with text gradient descent which stores bias in final array position - TODO normalize
        bias = coefficients[0];
        theta = Arrays.copyOfRange(coefficients, 1, coefficients.length);
        this.vocabulary = null;
    }

    /**
     * Returns most probable class for message consisting of <em>words</em>.
     *
     * @param words to classify
     * @return most probable class for message consisting of <em>words</em>
     * @throws IllegalArgumentException if words are null or empty
     */
    @Override
    public double classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        return Activation.SIGMOID.apply(bias + dotProduct(words));
    }

    /**
     * Returns most probable class for message consisting of <em>words</em>.
     *
     * @param explanatory features to classify
     * @return most probable class for message consisting of <em>words</em>
     * @throws IllegalArgumentException if words are null or empty
     */
    @Override
    public double predict(double[] explanatory) {
        if (explanatory == null || explanatory.length == 0) {
            throw new IllegalArgumentException("explanatory features must not be null or empty");
        }
        return Activation.SIGMOID.apply(bias + Vector.dotProduct(explanatory, theta));
    }

    // calculates dot product of words tf-idf and theta coefficients for associated words
    private double dotProduct(String[] words) {
        double sum = 0;
        for (TF_IDF_Term term : TF_IDF_Vectorizer.tfIdf(words, vocabulary)) {
            sum += term.getTfIdf() * theta[term.getId()];
        }
        return sum;
    }

}
