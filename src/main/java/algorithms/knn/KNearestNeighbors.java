package algorithms.knn;

import algorithms.model.Model;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import structures.Sample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class implements KNN algorithm that can be used for regression or classification. Inference runs in time proportional
 * to the size of training set so the algorithm is not suitable for large data sets.
 * <p>
 * The user is able to specify custom distance function or use one of the predefined functions in {@link DistanceFunction}.
 * </p>
 * Finally, there is a set of predefined aggregation functions that can be selected with {@link AggregationStrategy} which turn
 * KNN in standard or weighted form.
 *
 * @author dtemraz
 */
public class KNearestNeighbors implements Model {

    private final int k;
    private final List<Sample> dataSet;
    private final DistanceFunction distanceFunction;
    private final AggregateFunction aggregateFunction;

    public KNearestNeighbors(int k, List<Sample> dataSet, DistanceFunction distanceFunction, AggregationStrategy aggregationStrategy) {
        this.k = k;
        this.dataSet = dataSet;
        this.distanceFunction = distanceFunction;
        aggregateFunction = getAggregateFunction(aggregationStrategy);
    }

    @Override
    public double predict(double[] data) {
        List<Neighbor> neighbors = new ArrayList<>();
        for (Sample sample : dataSet) {
            double distance = distanceFunction.apply(data, sample.getValues());
            neighbors.add(new Neighbor(distance, sample.getTarget()));
        }
        Collections.sort(neighbors);
        return aggregateFunction.apply(neighbors);
    }

    /**
     * Returns number of nearest neighbors considered in inference.
     *
     * @return number of nearest neighbors considered in inference
     */
    public int getK() {
        return k;
    }

    // initializes aggregation function according to specified strategy
    private AggregateFunction getAggregateFunction(AggregationStrategy aggregationStrategy) {
        switch (aggregationStrategy) {
            case AVERAGE:
                return AggregateFunction.average(k);
            case INVERSE_DISTANCE_WEIGHTED_AVERAGE:
                return AggregateFunction.inverseDistanceWeightedAverage(k);
            case MAJORITY:
                return AggregateFunction.majority(k);
            case INVERSE_DISTANCE_WEIGHTED_MAJORITY:
                return AggregateFunction.inverseDistanceWeightedMajority(k);
            case INVERSE_CLASS_PROBABILITY_WEIGHTED_MAJORITY:
                return AggregateFunction.classProbabilityWeightedMajority(dataSet, k);
        }
        throw new IllegalStateException("unknown aggregation strategy: " + aggregationStrategy);
    }

    // this class represents neighbor to the query point
    @RequiredArgsConstructor
    @Getter
    static class Neighbor implements Comparable<Neighbor> {
        final double distance;
        final double target;

        @Override
        public int compareTo(Neighbor other) {
            return Double.compare(distance, other.distance);
        }
    }

}