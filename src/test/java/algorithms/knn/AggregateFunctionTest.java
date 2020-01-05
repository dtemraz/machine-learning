package algorithms.knn;

import org.junit.Test;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * This class tests {@link AggregateFunction} used to generate output value in {@link KNearestNeighbors}.
 * There are prepared lists of "nearest" neighbors {@link #regressionNeighbors} and {@link #classificationNeighbors}
 * to the imaginary query point.
 *
 * Each neighbor has hardcoded notion of distance to the query point which itself is not specified in the tests.
 */
public class AggregateFunctionTest {

    private static final int CLASS_ZERO = 0;
    private static final int CLASS_ONE = 1;
    private static final double delta = 0.0000000001;
    private static final int K = 2;
    private static final List<KNearestNeighbors.Neighbor> regressionNeighbors =
            getSamples(Map.of(1, 10, 2, 20, 10, 100, 20, 200, 30, 300, 40, 400, 50, 500));
    private static final List<KNearestNeighbors.Neighbor> classificationNeighbors =
            getSamples(Map.of(10, CLASS_ZERO, 20, CLASS_ZERO, 30, CLASS_ZERO, 1, CLASS_ONE, 2, CLASS_ONE));

    static {
        regressionNeighbors.sort(Comparator.comparing(KNearestNeighbors.Neighbor::getDistance));
        classificationNeighbors.sort(Comparator.comparing(KNearestNeighbors.Neighbor::getDistance));
    }

    // regression

    @Test
    public void average() {
        // given
        AggregateFunction average = AggregateFunction.average(K);
        // when
        double computed = average.apply(regressionNeighbors);
        // then
        assertEquals(sumTargets() / K, computed, delta);
    }

    @Test
    public void inverseDistanceWeightedAverage() {
        // given
        AggregateFunction weightedAverage = AggregateFunction.inverseDistanceWeightedAverage(K);
        // when
        double computed = weightedAverage.apply(regressionNeighbors);
        // then
        assertEquals(weightedAverage(), computed, delta);
    }

    // classification

    @Test
    public void majority() {
        // given
        AggregateFunction majority = AggregateFunction.majority(K);
        // when
        double computed = majority.apply(classificationNeighbors);
        // then
        assertEquals(CLASS_ONE, computed, delta);
    }

    /**
     * There are less samples of class 1 neighbors but they are closer to the query point and as such weighted favorably.
     */
    @Test
    public void inverseDistanceWeightedMajority() {
        // given
        AggregateFunction weightedMajority = AggregateFunction.inverseDistanceWeightedMajority(K);
        // when
        double computed = weightedMajority.apply(classificationNeighbors);
        // then
        assertEquals(CLASS_ONE, computed, delta);
    }


    private static List<KNearestNeighbors.Neighbor> getSamples(Map<Integer, Integer> samples) {
        return samples.entrySet().stream().map(e -> new KNearestNeighbors.Neighbor(e.getKey(), e.getValue())).collect(Collectors.toList());
    }

    private static double sumTargets() {
        return regressionNeighbors.subList(0, K).stream().map(KNearestNeighbors.Neighbor::getTarget).reduce(0D, Double::sum);
    }

    private static double weightedAverage() {
        double totalInverseDistance = sumInverseDistances();
        // weighted average each of target values with respect to their inverse distances
        return regressionNeighbors.subList(0, K).stream()
                .map(neighbor -> ((1 / neighbor.distance) / totalInverseDistance) * neighbor.target)
                .reduce(0D, Double::sum);
    }

    private static double sumInverseDistances() {
        return regressionNeighbors.subList(0, K).stream().map(neighbor -> 1 / neighbor.getDistance()).reduce(0D, Double::sum);
    }

}
