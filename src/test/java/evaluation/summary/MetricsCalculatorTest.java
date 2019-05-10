package evaluation.summary;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * This class tests that various classification metrics are correctly calculated.
 *
 */
public class MetricsCalculatorTest {

    private static final double A = 0.0;
    private static final double B = 1.0;
    private static final double C = 2.0;

    private static final double delta = 0.00000001;


    @Test
    public void precisionShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        // when
        Map<Double, Double> precision = MetricsCalculator.calculatePrecision(confusionMatrix);
        // then
        assertPrecisions(precision);
    }


    @Test
    public void recallShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        // when
        Map<Double, Double> recall = MetricsCalculator.calculateRecall(confusionMatrix);
        // then
        assertRecall(recall);
    }

    @Test
    public void accuracyShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        // when
        double accuracy = MetricsCalculator.calculateAccuracy(confusionMatrix);
        // then
        assertAccuracy(accuracy);
    }

    @Test
    public void f1ScoreShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        Map<Double, Double> precision = MetricsCalculator.calculatePrecision(confusionMatrix);
        Map<Double, Double> recall = MetricsCalculator.calculateRecall(confusionMatrix);
        // when
        double macroAvgF1 = MetricsCalculator.macroAvgF1(precision, recall);
        // then
        assertF1(macroAvgF1);
    }

    @Test
    public void macroAvgPrecisionShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        Map<Double, Double> precision = MetricsCalculator.calculatePrecision(confusionMatrix);
        // when
        double macroAvgPrecision = MetricsCalculator.macroAvgPrecision(precision);
        // then
        assertMacroPrecision(macroAvgPrecision);
    }

    @Test
    public void macroAvgRecallShouldBeCorrect() {
        // given
        Map<Double, Map<Double, Integer>> confusionMatrix = buildConfusionMatrix();
        Map<Double, Double> recall = MetricsCalculator.calculateRecall(confusionMatrix);
        // when
        double macroAvgRecall = MetricsCalculator.macroAvgRecall(recall);
        // then
        assertMacroRecall(macroAvgRecall);
    }


    private void assertPrecisions(Map<Double, Double> precision) {
        assertEquals(25D / (25 + 3 + 1), precision.get(A), delta);
        assertEquals(32D / (32 + 5 + 0), precision.get(B), delta);
        assertEquals(15D / (15 + 2 + 4), precision.get(C), delta);
    }

    private void assertMacroPrecision(double macroAvgPrecision) {
        double precisionA = 25D / (25 + 3 + 1);
        double precisionB = 32D / (32 + 5 + 0);
        double precisionC = 15D / (15 + 2 + 4);
        double expectedMacroAvg = (precisionA + precisionB + precisionC) / 3;
        assertEquals(expectedMacroAvg, macroAvgPrecision, delta);
    }

    private void assertRecall(Map<Double, Double> recall) {
        assertEquals(25D / (25 + 5 + 2), recall.get(A), delta);
        assertEquals(32D / (3 + 32 + 4), recall.get(B), delta);
        assertEquals(15D / (1 + 0 + 15), recall.get(C), delta);
    }


    private void assertMacroRecall(double macroAvgRecall) {
        double recallA = 25D / (25 + 5 + 2);
        double recallB = 32D / (3 + 32 + 4);
        double recallC = 15D / (1 + 0 + 15);
        double expectedMacroAvg = (recallA + recallB + recallC) / 3;
        assertEquals(expectedMacroAvg, macroAvgRecall, delta);
    }

    private void assertF1(double macroAvgF1) {
        double precisionA = 0.8620689655172413;
        double recallA = 0.78125;
        double f1A = 2 * precisionA * recallA / (precisionA + recallA);

        double precisionB = 0.8648648648648649;
        double recallB = 0.8205128205128205;
        double f1B = 2 * precisionB * recallB / (precisionB + recallB);

        double precisionC = 0.7142857142857143;
        double recallC = 0.9375;
        double f1C = 2 * precisionC * recallC / (precisionC + recallC);

        assertEquals((f1A + f1B + f1C) / 3, macroAvgF1, delta);
    }


    private void assertAccuracy(double accuracy) {
        double correct = 25 + 32 + 15;
        double total = (25 + 5 + 2) + (3 + 32 + 4) + (1 + 0 + 15);
        double expected = correct / total;
        assertEquals(expected, accuracy, delta);
    }

    private Map<Double, Map<Double, Integer>> buildConfusionMatrix() {
        Map<Double, Map<Double, Integer>> confusionMatrix = new LinkedHashMap<>();

        Map<Double, Integer> predictionsA = new LinkedHashMap<>();
        predictionsA.put(A, 25);
        predictionsA.put(B, 5);
        predictionsA.put(C, 2);
        confusionMatrix.put(A, predictionsA);

        Map<Double, Integer> predictionsB = new LinkedHashMap<>();
        predictionsB.put(A, 3);
        predictionsB.put(B, 32);
        predictionsB.put(C, 4);
        confusionMatrix.put(B, predictionsB);

        Map<Double, Integer> predictionsC = new LinkedHashMap<>();
        predictionsC.put(A, 1);
        predictionsC.put(B, 0);
        predictionsC.put(C, 15);
        confusionMatrix.put(C, predictionsC);

        return confusionMatrix;
    }

}
