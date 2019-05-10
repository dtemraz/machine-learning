package evaluation.summary;

import evaluation.StratifiedKFold;
import evaluation.StratifiedTrainAndTest;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeSet;

/**
 * This class defines a result of a {@link algorithms.model.TextModel} evaluation via {@link StratifiedTrainAndTest} and {@link StratifiedKFold}.
 * The class contains following metrics:
 * <ul>
 *  <li>overall accuracy over all classes</li>
 *  <li>confusion matrix</li>
 *  <li>precision per class</li>
 *  <li>recall per class</li>
 *  <li>macro-averaged F1 score</li>
 * </ul>
 *
 * @author dtemraz
 */
@Getter
@EqualsAndHashCode
@RequiredArgsConstructor
public class Summary {

    // overall accuracy over all classes, setter allows more precise average calculations
    private final double overallAccuracy;
    // how many times class was correctly classified or confused with the other class
    private final Map<Double, Map<Double, Integer>> confusionMatrix;
    // precision per each class
    private final Map<Double, Double> precision;
    // recall per each class
    private final Map<Double, Double> recall;
    // macro averaged f1 score
    private final double macroAvgF1;
    // macro averaged precision
    private final double macroAvgPrecision;
    // macro averaged recall
    private final double macroAvgRecall;

    // text, expected class and predicted class in a list as a separate entries
    private final Set<WronglyClassified> wronglyClassified;

    public Summary(Map<Double, Map<Double, Integer>> confusionMatrix, Set<WronglyClassified> wronglyClassified) {
        this.confusionMatrix = confusionMatrix;
        this.wronglyClassified = wronglyClassified;
        overallAccuracy = MetricsCalculator.calculateAccuracy(confusionMatrix);
        precision = MetricsCalculator.calculatePrecision(confusionMatrix);
        recall = MetricsCalculator.calculateRecall(confusionMatrix);
        macroAvgF1 = MetricsCalculator.macroAvgF1(precision, recall);
        macroAvgPrecision = MetricsCalculator.macroAvgPrecision(precision);
        macroAvgRecall = MetricsCalculator.macroAvgRecall(recall);
    }

    @Override
    public String toString() {
        return String.format("overall accuracy: %.4f\nprecision: %s\nrecall: %s\nmacro-avg F1: %.4f \nmacro-avg precision: %.4f\nmacro-avg recall: %.4f\n", overallAccuracy, precision, recall, macroAvgF1, macroAvgPrecision, macroAvgRecall) + formatConfusionMatrix();
    }

    // formats confusion matrix in a human readable format
    private String formatConfusionMatrix() {
        StringBuilder matrixFormatter = new StringBuilder();
        matrixFormatter.append("confusion matrix: \n");
        for (Map.Entry<Double, Map<Double, Integer>> entry : confusionMatrix.entrySet()) {
            Map<Double, Integer> confusions = entry.getValue();
            TreeSet<Double> orderedKeys = new TreeSet<>(confusions.keySet());
            String classHeader = entry.getKey() + " { ";
            matrixFormatter.append(classHeader);
            StringJoiner joiner = new StringJoiner(" , ");
            orderedKeys.forEach(key -> joiner.add(key + "=" + confusions.get(key)));
            matrixFormatter.append(joiner.toString());
            matrixFormatter.append("}\n");
        }
        return matrixFormatter.toString();
    }

    /**
     * Returns Set of wrongly classified samples as a List, where 0th element = text, 1th element = expected class and
     * 2nd element = predicted class.
     *
     * @return wrongly classified samples as a List, where 0th element = text, 1th element = expected class and
     * 2nd element = predicted class.
     */
    public Set<WronglyClassified> getWronglyClassified() {
        return wronglyClassified;
    }
}
