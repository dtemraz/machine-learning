package evaluation;

import evaluation.summary.Summary;
import evaluation.summary.SummaryAnalysis;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * This class test that average over summary objects is correctly calculated with {@link SummaryAnalysis#average(List)}.
 *
 * @author dtemraz
 */
public class SummaryAnalysisTest {

    private static final double CLASS_0 = 0.0;
    private static final double CLASS_1 = 1.0;

    private static final double delta = 0.00000001;

    @Test
    public void averageShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        assertCorrectAverage(average);
    }

    private List<Summary> getSummaries() {

        List<Summary> summaries = new ArrayList<>();

        /*
         * Build first summary instance
         */

        Map<Double, Double> classAccuracy = new HashMap<>();
        classAccuracy.put(CLASS_0, 0.92);
        classAccuracy.put(CLASS_1, 0.94);

        Map<Double, Map<Double, Integer>> confusionMatrix = new HashMap<>();

        Map<Double, Integer> confusion0 = new HashMap<>();
        confusion0.put(CLASS_0, 0);
        confusion0.put(CLASS_1, 10);

        Map<Double, Integer> confusion1 = new HashMap<>();
        confusion1.put(CLASS_1, 5);
        confusion1.put(CLASS_0, 5);

        confusionMatrix.put(CLASS_0, confusion0);
        confusionMatrix.put(CLASS_1, confusion1);

        summaries.add(new Summary(0.8, classAccuracy, confusionMatrix, new HashSet<>()));

        /*
         * Build second summary instance, refuse references - ok since this is a test
         */

        // reuse references for new Map objects
        classAccuracy = new HashMap<>();
        classAccuracy.put(CLASS_0, 0.90);
        classAccuracy.put(CLASS_1, 0.90);

        confusionMatrix = new HashMap<>();

        confusion0 = new HashMap<>();
        confusion0.put(CLASS_0, 0);
        confusion0.put(CLASS_1, 20);

        confusion1 = new HashMap<>();
        confusion1.put(CLASS_1, 10);
        confusion1.put(CLASS_0, 10);

        confusionMatrix.put(CLASS_0, confusion0);
        confusionMatrix.put(CLASS_1, confusion1);

        summaries.add(new Summary(0.88, classAccuracy, confusionMatrix, new HashSet<>()));
        return summaries;
    }

    private void assertCorrectAverage(Summary average) {
        // overall accuracy
        assertEquals((0.8 + 0.88) / 2, average.getOverallAccuracy(), delta);
        // class accuracy
        assertEquals((0.92 + 0.90) / 2, average.getClassAccuracy().get(CLASS_0), delta);
        assertEquals((0.94 + 0.90) / 2,  average.getClassAccuracy().get(CLASS_1), delta);
        // confusion matrix
        assertEquals((0 + 0) / 2, average.getConfusionMatrix().get(CLASS_0).get(CLASS_0), delta);
        assertEquals((10 + 20) / 2, average.getConfusionMatrix().get(CLASS_0).get(CLASS_1), delta);
        assertEquals((5 + 10) / 2, average.getConfusionMatrix().get(CLASS_1).get(CLASS_1), delta);
        assertEquals((5 + 10) / 2, average.getConfusionMatrix().get(CLASS_1).get(CLASS_0), delta);

    }

}
