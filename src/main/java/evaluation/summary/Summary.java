package evaluation.summary;

import evaluation.StratifiedKFold;
import evaluation.StratifiedTrainAndTest;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;

import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeSet;

/**
 * This class defines a result of a {@link algorithms.ensemble.model.TextModel} evaluation via {@link StratifiedTrainAndTest}
 * and {@link StratifiedKFold}.
 * The class contains following metrics:
 * <ul>
 *  <li>overall accuracy which is the accuracy over all classes</li>
 *  <li>class accuracy which is the accuracy per class</li>
 *  <li>confusion matrix which shows how many times class was correctly classified or confused with the other class</li>
 * </ul>
 *
 * @author dtemraz
 */
@Getter
@EqualsAndHashCode
public class Summary {

    // overall accuracy over all classes
    @Setter
    private double overallAccuracy;
    // accuracy per each class
    private final Map<Double, Double> classAccuracy;
    // how many times class was correctly classified or confused with the other class
    private final Map<Double, Map<Double, Integer>> confusionMatrix;
    // text, expected class and predicted class in a list as a separate entries
    private final Set<WronglyClassified> wronglyClassified; // Self-Note, refactor to generics to support non textual samples

    public Summary(double overallAccuracy, Map<Double, Double> classAccuracy, Map<Double, Map<Double, Integer>> confusionMatrix, Set<WronglyClassified> wronglyClassified) {
        this.overallAccuracy = overallAccuracy;
        this.classAccuracy = classAccuracy;
        this.confusionMatrix = confusionMatrix;
        this.wronglyClassified = wronglyClassified;
    }

    @Override
    public String toString() {
        return String.format("overall accuracy: %.4f \nper class accuracy: %s \n" , overallAccuracy, classAccuracy) + formatConfusionMatrix();
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
