package evaluation.summary;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class offers utility methods to perform analysis on list of {@link Summary} results from testing of machine learning models.
 * Currently, the only supported method is {@link #average(List)} which returns average values over a list of summaries.
 *
 * @author dtemraz
 */
public class SummaryAnalysis {

    /**
     * Returns average for overall accuracy, per class accuracy and confusion matrix for the <em>summaries</em> list.
     *
     * @param summaries to average
     * @return average for overall accuracy, per class accuracy and confusion matrix for the <em>summaries</em> list
     */
    public static Summary average(List<Summary> summaries) {
        return calculateAverage(summaries.size(), calculateTotals(summaries));
    }

    // returns summary object which contains sum of all metrics in the summaries list
    private static Summary calculateTotals(List<Summary> summaries) {
        double overallAccuracy = 0d;
        HashMap<Double, Double> perClassAccuracy = new HashMap<>();
        Map<Double, Map<Double, Integer>> confusionMatrix = new HashMap<>();
        Set<WronglyClassified> missedMessages = new HashSet<>(); // SET ensures no duplicates
        for (Summary summary : summaries) {
            overallAccuracy += summary.getOverallAccuracy();
            summary.getClassAccuracy().forEach((classId, accuracy) -> perClassAccuracy.merge(classId, accuracy, (old, n) -> old + n));
            summary.getConfusionMatrix().forEach((classId, matrix) -> {
                Map<Double, Integer> confusion = confusionMatrix.putIfAbsent(classId, matrix);
                if (confusion != null) {
                    matrix.forEach((k, v) -> confusion.merge(k, v, (old, n) -> old + n));
                }
            });
            missedMessages.addAll(summary.getWronglyClassified());
        }
        return new Summary(overallAccuracy, perClassAccuracy, confusionMatrix, missedMessages);
    }

    // divides all the metrics by the number of summaries to calculate averages
    private static Summary calculateAverage(int summaries, Summary totalSummary) {
        totalSummary.setOverallAccuracy(totalSummary.getOverallAccuracy() / summaries);
        totalSummary.getClassAccuracy().replaceAll((k, v) -> v / summaries);
        totalSummary.getConfusionMatrix().forEach((classId, matrix) -> matrix.replaceAll((k, v) -> v / summaries));
        return totalSummary;
    }

}
