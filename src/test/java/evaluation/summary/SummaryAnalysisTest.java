package evaluation.summary;

import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * This class test that average over summary objects is correctly calculated with {@link SummaryAnalysis#average(List)}.
 * <p>
 * Note that the numbers in tests are hardcoded to confusion matrices built with {@link SummaryAnalysisTest#getSummaries()}.
 * While i am generally not fond of this approach, tests for these metrics are actually more readable and easier to verify without any abstraction.
 * </p>
 *
 * @author dtemraz
 */
public class SummaryAnalysisTest {

    private static final double CLASS_0 = 0.0;
    private static final double CLASS_1 = 1.0;

    private static final double delta = 0.00000001;

    @Test
    public void averageAccuracyShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        assertAccuracy(average);
    }

    @Test
    public void averageConfusionMatrixShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        assertConfusionMatrix(average);
    }

    @Test
    public void averagePrecisionShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        assertPrecision(average);
    }

    @Test
    public void averageRecallShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        assertRecall(average);
    }

    @Test
    public void macroAverageF1ShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        double expected = summaries.stream().mapToDouble(Summary::getMacroAvgF1).reduce(0, Double::sum) / summaries.size();
        assertEquals(expected, average.getMacroAvgF1(), delta);
    }

    @Test
    public void macroAveragePrecisionShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        double expected = summaries.stream().mapToDouble(Summary::getMacroAvgPrecision).reduce(0, Double::sum) / summaries.size();
        assertEquals(expected, average.getMacroAvgPrecision(), delta);
    }

    @Test
    public void macroAverageRecallShouldBeCorrect() {
        // given
        List<Summary> summaries = getSummaries();
        // when
        Summary average = SummaryAnalysis.average(summaries);
        // then
        double expected = summaries.stream().mapToDouble(Summary::getMacroAvgRecall).reduce(0, Double::sum) / summaries.size();
        assertEquals(expected, average.getMacroAvgRecall(), delta);
    }

    private void assertAccuracy(Summary average) {
        // overall accuracy
        double accuracy1 = (double) (50 + 40) / (50 + 40 + 10 + 5);
        double accuracy2 = (double) (30 + 45) / (30 + 45 + 5 + 2);
        assertEquals((accuracy1 + accuracy2) / 2, average.getOverallAccuracy(), delta);
    }

    private void assertConfusionMatrix(Summary average) {
        // confusion matrix uses Integers to represent samples hence division bellow is ok
        assertEquals((50 + 30) / 2, average.getConfusionMatrix().get(CLASS_0).get(CLASS_0), delta);
        assertEquals((10 + 5) / 2, average.getConfusionMatrix().get(CLASS_0).get(CLASS_1), delta);
        assertEquals((40 + 45) / 2, average.getConfusionMatrix().get(CLASS_1).get(CLASS_1), delta);
        assertEquals((5 + 2) / 2, average.getConfusionMatrix().get(CLASS_1).get(CLASS_0), delta);
    }

    private void assertPrecision(Summary average) {
        // precision, do NOT refactor D otherwise this will be integer division
        double precision0 = ((50D / (50 + 5)) + (30D / (30 + 2))) / 2;
        assertEquals(precision0, average.getPrecision().get(CLASS_0), delta);
        double precision1 = ((40D / (40 + 10)) + (45D / (45 + 5))) / 2;
        assertEquals(precision1, average.getPrecision().get(CLASS_1), delta);
    }

    private void assertRecall(Summary average) {
        // recall, do not refactor D otherwise this will be integer division
        double recall0 = ((50D / (50 + 10)) + (30D / (30 + 5))) / 2;
        assertEquals(recall0, average.getRecall().get(CLASS_0), delta);
        double recall1 = ((40D / (40 + 5)) + (45D / (45 + 2))) / 2;
        assertEquals(recall1, average.getRecall().get(CLASS_1), delta);
    }


    private List<Summary> getSummaries() {

        List<Summary> summaries = new ArrayList<>();

        /*
         Build first summary instance:
         predicted ->
           0  1
         0 50 10
         1 5  40
         */

        Map<Double, Map<Double, Integer>> confusionMatrix = new LinkedHashMap<>();

        Map<Double, Integer> confusion0 = new LinkedHashMap<>();
        confusion0.put(CLASS_0, 50);
        confusion0.put(CLASS_1, 10);

        Map<Double, Integer> confusion1 = new LinkedHashMap<>();
        confusion1.put(CLASS_0, 5);
        confusion1.put(CLASS_1, 40);

        confusionMatrix.put(CLASS_0, confusion0);
        confusionMatrix.put(CLASS_1, confusion1);

        summaries.add(new Summary(confusionMatrix, new HashSet<>()));

        /*
         Build second summary instance:
         predicted ->
           0  1
         0 30 5
         1 2  45
         */

        confusionMatrix = new LinkedHashMap<>();

        confusion0 = new LinkedHashMap<>();
        confusion0.put(CLASS_0, 30);
        confusion0.put(CLASS_1, 5);

        confusion1 = new LinkedHashMap<>();
        confusion1.put(CLASS_0, 2);
        confusion1.put(CLASS_1, 45);

        confusionMatrix.put(CLASS_0, confusion0);
        confusionMatrix.put(CLASS_1, confusion1);

        summaries.add(new Summary(confusionMatrix, new HashSet<>()));
        return summaries;
    }

}
