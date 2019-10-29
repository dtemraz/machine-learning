package algorithms.linear_regression;

import algorithms.linear_regression.optimization.real_vector.Optimizer;
import algorithms.linear_regression.optimization.text.TextOptimizer;
import algorithms.model.ClassificationResult;
import algorithms.model.Model;
import algorithms.model.TextModel;
import algorithms.model.TextModelWithProbability;
import structures.text.Vocabulary;

import java.util.*;
import java.util.stream.Collectors;

/**
 * This class implements One-Against-Rest multi-class classifier. The classifier can be initialized to support textual
 * classification with factory method {@link #getTextModel(Vocabulary, Map, TextOptimizer)}.
 *
 * <p>
 * The class creates a single instance of {@link LogisticRegression} per target class and specializes each instance to
 * recognize one class as the true class, and all others as 'false' classes. When performing classification of data, class associated
 * with logistic regression instance that outputs highest probability is chosen as the correct class.
 * </p>
 *
 * @author dtemraz
 */
public class OneAgainstRest implements TextModelWithProbability, Model {

    private static final double TARGET = 1D; // predictors are trained to recognize exactly one class as a true class
    private static final double OTHERS = 0D; // predictors are trained to recognize all other classes as a 'false' class

    // instances of logistic regression specialized to classify a single class defined with the key
    private final HashMap<Double, LogisticRegression> predictors = new HashMap<>();


    /**
     * Returns {@link OneAgainstRest} instance which can be used to classify text into multiple classes.
     *
     * @param vocabulary  which defines possible words and their indexes
     * @param trainingSet map of classes and messages broken into words per classes
     * @param optimizer   optimizer instance to train classifiers with gradient descent configuration
     * @return OneAgainstRest instance which can be used to classify text
     */
    public static TextModel getTextModel(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        return new OneAgainstRest(vocabulary, trainingSet, optimizer);
    }

    // initializes and trains instance of logistic regression per class for textual classification
    private OneAgainstRest(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        trainingSet.keySet().forEach(key -> {
            double targetClass = key;
            LogisticRegression regression = initialize(vocabulary, trainingSet, targetClass, optimizer);
            predictors.put(targetClass, regression);
        });
    }

    /**
     * Returns {@link OneAgainstRest} instance which can be used to classify text into multiple classes.
     *
     * @param trainingSet map of classes and messages broken into words per classes
     * @param optimizer   optimizer instance to train classifiers with gradient descent configuration
     * @return OneAgainstRest instance which can be used to classify text
     */
    public static Model getModel(List<double[]> trainingSet, Optimizer optimizer) {
        return new OneAgainstRest(trainingSet, optimizer);
    }

    // initializes and trains instance of logistic regression per class for classification over real number vectors
    private OneAgainstRest(List<double[]> trainingSet, Optimizer optimizer) {
        int classIndex = trainingSet.get(0).length - 1;
        Set<Double> classIds = trainingSet.stream().map(sample -> sample[classIndex]).collect(Collectors.toSet());
        classIds.forEach(classId -> {
            LogisticRegression regression = initialize(trainingSet, classId, optimizer);
            predictors.put(classId, regression);
        });
    }

    @Override
    public double classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        return maxProbabilityClass(words);
    }

    @Override
    public ClassificationResult classifyWithProb(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }

        return classProbabilities(words);
    }

    /**
     * Returns most probable class for <em>data</em>.
     *
     * @param data vector to classify
     * @return most probable class for <em>data</em>
     * @throws IllegalArgumentException if data is null or empty
     */
    @Override
    public double predict(double[] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be null or empty");
        }
        return maxProbabilityClass(data);
    }

    // chooses class associated with logistic regression model that outputs highest probability for words
    private double maxProbabilityClass(String[] words) {
        double max = Double.NEGATIVE_INFINITY;
        double targetClass = -1;
        // find max algorithm
        for (Map.Entry<Double, LogisticRegression> predictor : predictors.entrySet()) {
            double prediction = predictor.getValue().classify(words);
            if (prediction > max) {
                max = prediction;
                targetClass = predictor.getKey();
            }
        }
        return targetClass;
    }

    // computes probabilities of each class label, note that sum will not be normalized to 1 unlike with SoftMax regression
    private ClassificationResult classProbabilities(String[] words) {
        double max = Double.NEGATIVE_INFINITY;
        double targetClass = -1;
        double[] probabilities = new double[predictors.keySet().size()];

        // find max algorithm
        int label = 0;
        for (Map.Entry<Double, LogisticRegression> predictor : predictors.entrySet()) {
            Double prediction = predictor.getValue().classify(words);
            if (prediction > max) {
                max = prediction;
                targetClass = predictor.getKey();
            }
            probabilities[label++] = prediction;
        }
        return new ClassificationResult(targetClass, max, probabilities);
    }

    // chooses class associated with logistic regression model that outputs highest probability for features
    private double maxProbabilityClass(double[] features) {
        return predictors.entrySet().stream().max(Comparator.comparingDouble(e -> e.getValue().predict(features))).get().getKey();
    }


    // creates logistic regression instance from training set, trained to recognize target class among all others
    private LogisticRegression initialize(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, Double targetClass, TextOptimizer optimizer) {
        List<String[]> targetSamples = new ArrayList<>(trainingSet.get(targetClass)); // all samples belonging to target class
        List<String[]> rest = trainingSet.entrySet().stream() // samples belonging to all other classes
                .filter(e -> !e.getKey().equals(targetClass))
                .flatMap(e -> e.getValue().stream())
                .collect(Collectors.toList());

        // each logistic regression instance is trained to recognize specific class as 'true' class, and every other as 'false'
        Map<Double, List<String[]>> instanceSet = new HashMap<>();
        instanceSet.put(TARGET, targetSamples);
        instanceSet.put(OTHERS, rest);

        return new LogisticRegression(vocabulary, instanceSet, optimizer);
    }

    // creates logistic regression instance from training set, trained to recognize target class among all others
    private LogisticRegression initialize(List<double[]> trainingSet, Double targetClass, Optimizer optimizer) {
        ArrayList<double[]> samples = new ArrayList<>(trainingSet);
        int classIndex = samples.get(0).length - 1;
        for (double[] sample : samples) {
            if (sample[classIndex] == targetClass) {
                sample[classIndex] = TARGET;
            } else {
                sample[classIndex] = OTHERS;
            }
        }
        return new LogisticRegression(samples, optimizer);
    }

}