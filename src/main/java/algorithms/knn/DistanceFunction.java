package algorithms.knn;

import java.io.Serializable;

/**
 * This interface defines a distance function between two vectors. There is implementation of most common distance metrics:
 * <ul>
 * <li>squared euclidean</li>
 * <li>euclidean</li>
 * <li>manhattan</li>
 * </ul>
 * that the user may chose from.
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface DistanceFunction extends Serializable {

    /**
     * Returns distance between vectors <em>v1</em> and <em>v2</em>. Vectors <strong>must</strong> be of the same length.
     *
     * @param v1 vector for which to compute distance
     * to v2
     * @param v2 vector for which to compute distance to v1
     * @return distance between vectors <em>v1</em> and <em>v2</em>
     * @throws IllegalArgumentException if <em>v1</em> and <em>v2</em> are od different length
     */
    double apply(double[] v1, double[] v2);

    DistanceFunction SQUARED_EUCLIDEAN = (v1, v2) -> {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException(String.format("v1 and v2 of different length, respectively: %d and %d", v1.length, v2.length));
        }
        int length = v1.length;
        double distance = 0;
        for (int component = 0; component < length; component++) {
            double delta = v1[component] - v2[component];
            distance += delta * delta;
        }
        return distance;
    };

    DistanceFunction EUCLIDEAN = (v1, v2) -> Math.sqrt(SQUARED_EUCLIDEAN.apply(v1, v2));

    DistanceFunction MANHATTAN = (v1, v2) -> {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException(String.format("v1 and v2 of different length, respectively: %d and %d", v1.length, v2.length));
        }
        int length = v1.length;
        double distance = 0;
        for (int component = 0; component < length; component++) {
            distance += Math.abs(v1[component] - v2[component]);
        }
        return distance;
    };


}
