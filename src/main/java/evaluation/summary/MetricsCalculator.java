package evaluation.summary;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * This class lets user compute common machine learning metrics from confusion matrix: <em>accuracy</em>, <em>precision</em>, <em>recall</em>, <em>macro-averaged F1 score</em>,
 *  <em>macro-averaged precision</em> and  <em>macro-averaged recall</em>.
 *
 * @author dtemraz
 */
class MetricsCalculator {

    /**
     * Returns <strong>overall accuracy</strong> value from the <em>confusionMatrix</em>.
     *
     * @param confusionMatrix from which to compute overall accuracy
     * @return <strong>overall accuracy</strong> value from the <em>confusionMatrix</em>
     */
    static double calculateAccuracy(Map<Double, Map<Double, Integer>> confusionMatrix) {
        double correct = 0; // total number of correct predictions across all classes
        double total = 0; // total number of samples in all classes
        for (Map.Entry<Double, Map<Double, Integer>> entry : confusionMatrix.entrySet()) {
            double actual = entry.getKey();
            Map<Double, Integer> predictions = entry.getValue();
            for (Map.Entry<Double, Integer> prediction : predictions.entrySet()) {
                double predicted = prediction.getKey();
                double samples = prediction.getValue();
                if (predicted == actual) {
                    correct += samples;
                }
                total += samples;
            }
        }
        return correct / total;
    }

    /**
     * Returns macro-averaged F1 score from <em>precision</em> and <em>recall</em> values for each class.
     *
     * @param classPrecision table of precision per class
     * @param classRecall    table of recall per class
     * @return macro-averaged F1 score from <em>precision</em> and <em>recall</em> values for each class
     */
    static double macroAvgF1(Map<Double, Double> classPrecision, Map<Double, Double> classRecall) {
        double f1 = 0;
        Set<Double> classes = classPrecision.keySet(); // could have used recall classes as well
        for (Double classId : classes) {
            double precision = classPrecision.get(classId);
            double recall = classRecall.get(classId);
            f1 += (2 * precision * recall) / (precision + recall);
        }
        return f1 / classes.size();
    }

    /**
     * Returns macro averaged precision from <em>classPrecision</em> map.
     *
     * @param classPrecision map from which to compute macro averaged precision
     * @return Returns macro averaged precision from <em>classPrecision</em> map
     */
    static double macroAvgPrecision(Map<Double, Double> classPrecision) {
        return classPrecision.values().stream().reduce(0D, Double::sum) / classPrecision.size();
    }

    /**
     * Returns macro averaged recall from <em>classRecall</em> map.
     *
     * @param classRecall map from which to compute macro averaged precision
     * @return Returns macro averaged precision from <em>classRecall</em> map
     */
    static double macroAvgRecall(Map<Double, Double> classRecall) {
        return classRecall.values().stream().reduce(0D, Double::sum) / classRecall.size();
    }

    /**
     * Returns <strong>precision</strong> value per each class in <em>confusionMatrix</em>.
     *
     * @param confusionMatrix from which to compute precision
     * @return <strong>precision</strong> value per each class in <em>confusionMatrix</em>
     */
    static Map<Double, Double> calculatePrecision(Map<Double, Map<Double, Integer>> confusionMatrix) {
        Map<Double, Double> classPrecision = new LinkedHashMap<>();
        // prepare keys in advance to maintain same order in precision map, merge function will permute order
        confusionMatrix.keySet().forEach(k -> classPrecision.put(k, 0D));
        // in first step calculate all false positives for a class across other classes
        for (Map.Entry<Double, Map<Double, Integer>> entry : confusionMatrix.entrySet()) {
            Double actualClass = entry.getKey();
            // a single row od predictions for actual class
            for (Map.Entry<Double, Integer> prediction : entry.getValue().entrySet()) {
                // FP = sum of all values in column except TP
                Double currentClass = prediction.getKey();
                if (actualClass.equals(currentClass)) {
                    continue;
                }
                classPrecision.merge(currentClass, (double) prediction.getValue(), Double::sum);
            }
        }

        // with computed false positives it is easy to calculate precision as true positive can be directly read for each class
        for (Map.Entry<Double, Double> prediction : classPrecision.entrySet()) {
            Double actualClass = prediction.getKey();
            Integer truePositive = confusionMatrix.get(actualClass).get(actualClass);
            Double falsePositive = prediction.getValue();
            prediction.setValue(truePositive / (truePositive + falsePositive));
        }
        return classPrecision;
    }

    /**
     * Returns <strong>recall</strong> value per each class in <em>confusionMatrix</em>.
     *
     * @param confusionMatrix from which to compute recall
     * @return <strong>recall</strong> value per each class in <em>confusionMatrix</em>
     */
    static Map<Double, Double> calculateRecall(Map<Double, Map<Double, Integer>> confusionMatrix) {
        Map<Double, Double> classRecall = new LinkedHashMap<>();
        // prepare keys in advance to maintain same order in recall map, merge function will permute order
        confusionMatrix.keySet().forEach(k -> classRecall.put(k, 0D));
        for (Map.Entry<Double, Map<Double, Integer>> entry : confusionMatrix.entrySet()) {
            Double classId = entry.getKey();
            Map<Double, Integer> predictions = entry.getValue();
            Integer truePositive = predictions.get(classId);
            // FN = sum of all values in row except TP
            Integer falseNegative = (predictions.values().stream().reduce(0, Integer::sum)) - truePositive;
            double precision = truePositive / (double) (truePositive + falseNegative);
            classRecall.put(classId, precision);
        }
        return classRecall;
    }

}
