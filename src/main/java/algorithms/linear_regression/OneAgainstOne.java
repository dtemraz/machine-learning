package algorithms.linear_regression;

import algorithms.WithThreshold;
import algorithms.linear_regression.optimization.real_vector.Optimizer;
import algorithms.linear_regression.optimization.text.TextOptimizer;
import algorithms.model.Model;
import algorithms.model.TextModel;
import lombok.AccessLevel;
import lombok.EqualsAndHashCode;
import lombok.RequiredArgsConstructor;
import structures.text.Vocabulary;
import utilities.math.Statistics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class implements One-Against-One multi-class classifier. The classifier can be initialized to support textual
 * classification with factory method {@link #getTextModel(Vocabulary, Map, TextOptimizer)} or with {@link #getModel(List, Optimizer)} for real vectors classification.
 * <p>
 * The class creates a single instance of {@link LogisticRegression} for each combination of class pairs. When performing classification of data, class with most
 * votes across all predictor pairs is chosen as a winner.
 * The obvious problem is that number of pairs is equal to ((n-1) * n) / 2 where n is a number of classes.
 * </p>
 *   n  {@literal =>} n! / (2!(n-2)!) = (n-2)!*(n-1)*(n) / (2*(n-2)!)
 *   2
 * @author dtemraz
 */
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public class OneAgainstOne implements TextModel, Model {

    // fit binary classifier for each pair to recognize single target class
    private static final double TARGET = 1D;
    private static final double OTHER = 0D;
    private static final double THRESHOLD = 0.5D;

    // predictor pair for each combination
    private final Map<Pair, TextModel> textPredictors;
    private final Map<Pair, Model> predictors;

    /**
     * Returns {@link OneAgainstOne} instance which can be used to classify text into multiple classes. Predictor is built for each combination of class pairs.
     *
     * @param vocabulary  which defines possible words and their indexes
     * @param trainingSet map of classes and messages broken into words per classes
     * @param optimizer   optimizer instance to train classifiers with gradient descent configuration
     * @return OneAgainstOne instance which can be used to classify text
     */
    public static TextModel getTextModel(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        return initializePredictors(vocabulary, trainingSet, optimizer);
    }

    /**
     * Returns {@link OneAgainstOne} instance which can be used to classify data into multiple classes. Predictor is built for each combination of class pairs.
     *
     * @param trainingSet list of learning samples, class id is encoded in final component of sample vector
     * @param optimizer   optimizer instance to train classifiers with gradient descent configuration
     * @return OneAgainstRest instance which can be used to classify text
     */
    public static Model getModel(List<double[]> trainingSet, Optimizer optimizer) {
        return initializePredictors(trainingSet, optimizer);
    }

    private static OneAgainstOne initializePredictors(Vocabulary vocabulary, Map<Double, List<String[]>> trainingSet, TextOptimizer optimizer) {
        Map<Pair, TextModel> predictors = new HashMap<>();
        Double[] classes = trainingSet.keySet().toArray(new Double[trainingSet.size()]);
        // iterate over all class pair combinations
        for (int outer = 0; outer < classes.length - 1; outer++) {
            for (int inner = outer + 1; inner < classes.length; inner++) {
                Double target = classes[outer];
                Double other = classes[inner];
                // generate classifier for each pair
                Map<Double, List<String[]>> localSet = new HashMap<>();
                localSet.put(TARGET, trainingSet.get(target));
                localSet.put(OTHER, trainingSet.get(other));
                predictors.put(new Pair(target, other), WithThreshold.textModel(new LogisticRegression(vocabulary, localSet, optimizer), THRESHOLD));
            }
        }
        return new OneAgainstOne(predictors, null);
    }

    private static OneAgainstOne initializePredictors(List<double[]> trainingSet, Optimizer optimizer) {
        Map<Pair, Model> predictors = new HashMap<>();
        int classIndex = trainingSet.get(0).length - 1;
        // iterate over all class pair combinations
        for (int outer = 0; outer < trainingSet.size() - 1; outer++) {
            for (int inner = outer + 1; inner < trainingSet.size(); inner++) {
                double[] target = trainingSet.get(outer);
                double[] other = trainingSet.get(inner);
                // generate classifier for each pair
                List<double[]> localSet = new ArrayList<>();
                localSet.add(target);
                localSet.add(other);
                predictors.put(new Pair(target[classIndex], other[classIndex]), WithThreshold.model(new LogisticRegression(localSet, optimizer), THRESHOLD));
            }
        }
        return new OneAgainstOne(null, predictors);
    }

    @Override
    public double classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        double[] predictions = textPredictors.entrySet().stream()
                .mapToDouble(e -> e.getValue().classify(words) == TARGET ? e.getKey().targetClass : e.getKey().otherClass).toArray();
        return Statistics.mode(predictions);
    }

    @Override
    public double predict(double[] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be null or empty");
        }
        double[] predictions = predictors.entrySet().stream()
                .mapToDouble(e -> e.getValue().predict(data) == TARGET ? e.getKey().targetClass : e.getKey().otherClass).toArray();
        return Statistics.mode(predictions);
    }

    @RequiredArgsConstructor(access = AccessLevel.PRIVATE)
    @EqualsAndHashCode
    private static class Pair {
        private final double targetClass;
        private final double otherClass;
    }

}