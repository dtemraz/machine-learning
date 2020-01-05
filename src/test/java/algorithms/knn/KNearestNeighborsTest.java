package algorithms.knn;

import org.junit.Before;
import org.junit.Test;
import structures.Sample;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class KNearestNeighborsTest {

    private static final int POSITIVE = 1;
    private static final int NEGATIVE = -1;
    private static final int K = 4;
    private static final double EXACT_MATCH = 0;

    private List<Sample> samples;

    @Before
    public void setUp() {
        samples = defaultSetup();
    }

    /**
     * Tests that vectors are correctly classified according to the majority voting role, [2,2] should have more positives
     * neighbors while [5,5] should have more negative class neighbors.
     */
    @Test
    public void majorityClassification() {
        // given
        KNearestNeighbors knn = new KNearestNeighbors(K, samples, DistanceFunction.SQUARED_EUCLIDEAN, AggregationStrategy.MAJORITY);
        // when
        double expectedPositive = knn.predict(new double[]{2, 2});
        double expectedNegative = knn.predict(new double[]{5, 5});
        // then
        assertEquals(POSITIVE, expectedPositive, EXACT_MATCH);
        assertEquals(NEGATIVE, expectedNegative, EXACT_MATCH);
    }

    /**
     * Test that less common class is selected if there are 4 <strong>equidistant</strong> points to the query point, 2 from each class.
     * If a class is more common, it should have more samples supporting the query point.
     */
    @Test
    public void inverseProbabilityWeightedClassification() {
        // given
        KNearestNeighbors knn = new KNearestNeighbors(K, samples, DistanceFunction.SQUARED_EUCLIDEAN, AggregationStrategy.INVERSE_CLASS_PROBABILITY_WEIGHTED_MAJORITY);
        // when
        double expectNegative = knn.predict(new double[]{2.5, 2.5});
        // then
        assertEquals(POSITIVE, expectNegative, EXACT_MATCH);
    }

    private List<Sample> defaultSetup() {
        List<Sample> samples = new ArrayList<>();
        samples.add(new Sample(new double[]{1, 1}, POSITIVE));
        samples.add(new Sample(new double[]{2, 1}, POSITIVE));
        samples.add(new Sample(new double[]{3, 1}, POSITIVE));
        samples.add(new Sample(new double[]{4, 1}, POSITIVE));

        samples.add(new Sample(new double[]{1, 4}, NEGATIVE));
        samples.add(new Sample(new double[]{2, 4}, NEGATIVE));
        samples.add(new Sample(new double[]{3, 4}, NEGATIVE));
        samples.add(new Sample(new double[]{4, 4}, NEGATIVE));
        samples.add(new Sample(new double[]{5, 4}, NEGATIVE));
        return samples;
    }


}
