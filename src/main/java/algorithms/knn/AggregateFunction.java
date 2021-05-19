package algorithms.knn;

import structures.Sample;
import utilities.math.Statistics;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * This interface defines set of aggregation functions that can be used with KNN in regression or classification setting.
 *
 * Functions {@link #average(int)} and {@link #majority(int)} are non weighted, functions {@link #inverseDistanceWeightedAverage(int)}
 * and {@link #inverseDistanceWeightedMajority(int)} assign weights to each member proportional to how close a neighbor is to
 * a query point and function {@link #classProbabilityWeightedMajority(List, int)} assign weight proportional to probability
 * of a neighbor class as observed in training set.
 *
 * @author dtemraz
 */
@FunctionalInterface
interface AggregateFunction extends Serializable {

    /**
     * Returns aggregation of values of closest neighbors. The result could be classification or regression, depending on the
     * implementation of this function.
     * <p>
     * The method expects that <em>neighbors</em> are in <strong>descending sorted</strong> order, closer ones appearing
     * first in the list.
     * </p>
     *
     * @param neighbors in sorted descending order to the query point
     * @return aggregation of closest neighbor values
     */
    double apply(List<KNearestNeighbors.Neighbor> neighbors);

    // REGRESSION

    // average of k closest neighbors, regardless of their distance to the query point
    static AggregateFunction average(int k) {
        return neighbors -> Statistics.mean(kMostSimilar(neighbors, k));
    }

    // computes prediction as weighted average of k closest neighbors, weighted by inverse distance to the query point
    static AggregateFunction inverseDistanceWeightedAverage(int k) {
        return neighbors -> {
            double inverseDistanceSum = 0;
            for (int i = 0; i < k; i++) {
                KNearestNeighbors.Neighbor neighbor = neighbors.get(i);
                // exact match of a sample
                if (neighbor.distance == 0) {
                    return neighbor.target;
                }
                inverseDistanceSum += 1 / neighbor.distance;
            }
            return weightedAverage(k, neighbors, inverseDistanceSum);
        };
    }

    // helper method to compute weighted average
    private static double weightedAverage(int k, List<KNearestNeighbors.Neighbor> neighbors, double inverseDistanceSum) {
        double prediction = 0;
        for (int i = 0; i < k; i++) {
            KNearestNeighbors.Neighbor neighbor = neighbors.get(i);
            double inverseDistance = 1 / neighbor.distance;
            // normalize all inverse distances so they sum up to 1
            double w = inverseDistance / inverseDistanceSum;
            prediction += neighbor.target * w;
        }
        return prediction;
    }

    // CLASSIFICATION

    // most represented class among k closest neighbors, regardless of their distance to the query point
    static AggregateFunction majority(int k) {
        return neighbors -> Statistics.mode(kMostSimilar(neighbors, k));
    }

    // most represented class among k closest neighbors, weighted by their according to distance to the query point
    static AggregateFunction inverseDistanceWeightedMajority(int k) {
        return neighbors -> {
            double max = Double.NEGATIVE_INFINITY;
            double prediction = -1;
            HashMap<Double, Double> weightedMajority = new HashMap<>((int) (k / 0.75) + 1);
            for (int i = 0; i < k; i++) {
                KNearestNeighbors.Neighbor neighbor = neighbors.get(i);
                // exact match of a sample
                if (neighbor.distance == 0) {
                    return neighbor.target;
                }
                double inverseDistance = 1 / neighbor.distance;
                // maintain weight per class as sum of inverse distances for neighbors of that class
                Double weight = weightedMajority.merge(neighbor.target, inverseDistance, Double::sum);
                if (weight > max) {
                    prediction = neighbor.target;
                    max = weight;
                }
            }
            return prediction;
        };
    }

    // most represented class among k closest neighbors weighted by the inverse class probability
    static AggregateFunction classProbabilityWeightedMajority(List<Sample> dataSet, int k) {
        int total = dataSet.size();
        Map<Double, Double> classProbability = new HashMap<>();
        dataSet.forEach(sample -> classProbability.merge(sample.getTarget(), 1D / total, Double::sum));

        return neighbors -> {
            double max = Double.NEGATIVE_INFINITY;
            double label = -1;
            Map<Double, Double> weightedMajority = new HashMap<>((int) (k / 0.75) + 1);
            // weight each neighbor class according to inverse probability of the class, more common classes should appear more often
            for (double neighborClass : kMostSimilar(neighbors, k)) {
                Double classWeight = weightedMajority.merge(neighborClass, 1 / classProbability.get(neighborClass), Double::sum);
                if (classWeight > max) {
                    max = classWeight;
                    label = neighborClass;
                }
            }
            return label;
        };
    }

    // returns k nearest neighbors mapped to their target values
    private static double[] kMostSimilar(List<KNearestNeighbors.Neighbor> neighbors, int k) {
        double[] closestValues = new double[k];
        for (int neighbor = 0; neighbor < k; neighbor++) {
            closestValues[neighbor] = neighbors.get(neighbor).target;
        }
        return closestValues;
    }

}


