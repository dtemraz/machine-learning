package evaluation.summary;

import java.util.*;

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
        return calculateAverages(summaries);
    }

    // returns summary object which contains sum of all metrics in the summaries list
    private static Summary calculateAverages(List<Summary> summaries) {
        double overallAccuracy = 0d;
        double macroAvgF1 = 0d;
        double macroAvgPrecision = 0d;
        double macroAvgRecall = 0d;

        Map<Double, Map<Double, Integer>> confusionMatrix = new LinkedHashMap<>();
        Map<Double, Double> classPrecision = new LinkedHashMap<>();
        Map<Double, Double> classRecall = new LinkedHashMap<>();
        Set<WronglyClassified> missedMessages = new TreeSet<>();

        for (Summary summary : summaries) {
            // sum concrete metric values for each summary
            overallAccuracy += summary.getOverallAccuracy();
            macroAvgF1 += summary.getMacroAvgF1();
            macroAvgPrecision += summary.getMacroAvgPrecision();
            macroAvgRecall += summary.getMacroAvgRecall();

            // sum confusion matrix values for each summary
            summary.getConfusionMatrix().forEach((classId, matrix) -> {
                Map<Double, Integer> confusion = confusionMatrix.putIfAbsent(classId, matrix);
                if (confusion != null) {
                    matrix.forEach((k, v) -> confusion.merge(k, v, (old, delta) -> old + delta));
                }
            });

            // sum precision and recall values for each summary and save all missed messages
            summary.getPrecision().forEach((label, precision) -> classPrecision.merge(label, precision, Double::sum));
            summary.getRecall().forEach((label, recall) -> classRecall.merge(label, recall, Double::sum));
            missedMessages.addAll(summary.getWronglyClassified());
        }
        // compute macro averaged metrics
        final int n = summaries.size();
        overallAccuracy /= n;
        macroAvgF1 /= n;
        macroAvgPrecision /= n;
        macroAvgRecall /= n;

        // compute average values for matrix type metrics
        confusionMatrix.forEach((classId, matrix) -> matrix.replaceAll((k, v) -> v / n));
        classPrecision.entrySet().forEach(e -> e.setValue(e.getValue() / n));
        classRecall.entrySet().forEach(e -> e.setValue(e.getValue() / n));

        // could have computed metrics from average confusion matrix but this will be a little bit more precise for smaller data sets
        return new Summary(overallAccuracy, confusionMatrix, classPrecision, classRecall, macroAvgF1, macroAvgPrecision, macroAvgRecall, missedMessages);
    }

}
