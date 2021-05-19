package algorithms.knn;

import algorithms.model.Model;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import structures.Sample;
import utilities.QuickSelect;

import java.util.ArrayList;
import java.util.List;

/**
 * This class implements KNN algorithm that can be used for regression or classification. Inference runs in time proportional
 * to the size of training set so the algorithm is not suitable for large data sets.
 *
 * <p>
 * The user is able to specify custom distance function or use one of the predefined functions in {@link DistanceFunction}.
 * </p>
 * There is a set of predefined aggregation functions that can be selected with {@link AggregationStrategy} that apply
 * different neighbor weighting schemas.
 * <p>
 * User may also specify {@link SearchStrategy} used to find K nearest neighbors out of all neighbors, of those {@link SearchStrategy#QUICK_SELECT}
 * should be preferred in most if not all realistic scenarios.
 * </p>
 *
 * @author dtemraz
 */
public class KNearestNeighbors implements Model {

    private final int k;
    private final List<Sample> dataSet;
    private final DistanceFunction distanceFunction;
    private final AggregateFunction aggregateFunction;
    private final SearchStrategy searchStrategy;

    public KNearestNeighbors(int k, List<Sample> dataSet, DistanceFunction distanceFunction, AggregationStrategy aggregationStrategy) {
        this(k, dataSet, distanceFunction, aggregationStrategy, SearchStrategy.QUICK_SELECT);
    }

    public KNearestNeighbors(int k, List<Sample> dataSet, DistanceFunction distanceFunction, AggregationStrategy aggregationStrategy,
                             SearchStrategy searchStrategy) {
        this.k = k;
        this.dataSet = dataSet;
        this.distanceFunction = distanceFunction;
        this.aggregateFunction = getAggregateFunction(aggregationStrategy);
        this.searchStrategy = searchStrategy;
    }

    @Override
    public double predict(double[] data) {
        List<Neighbor> neighborDistances = computeNeighborDistances(data);
        List<Neighbor> kNearest = searchStrategy.findKNearest(neighborDistances, k);
        return aggregateFunction.apply(kNearest);
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

    private List<Neighbor> computeNeighborDistances(double[] data) {
        List<Neighbor> neighborDistances = new ArrayList<>();
        for (Sample sample : dataSet) {
            double distance = distanceFunction.apply(data, sample.getValues());
            neighborDistances.add(new Neighbor(distance, sample.getTarget()));
        }
        return neighborDistances;
    }

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