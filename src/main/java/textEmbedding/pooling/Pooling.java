package textEmbedding.pooling;

import java.io.Serializable;

/**
 * This interface defines pooling operation for (word) vectors. User may choose from pre-built options or provide
 * custom implementation. {@link Pooling#AVG} or {@link Pooling#MAX} should give best results most of the time.
 * Pooling is required for non-sequence models to ensure that model can work with texts with different number of words.
 */
public interface Pooling extends Serializable {

    /**
     * Returns vector, result of pooling operation(reduction) for <em>vectors</em>.
     *
     * @param vectors for which to apply pooling
     * @return vector, result of pooling operation(reduction) for <em>vectors</em>
     */
    double[] apply(double[]... vectors);

    /**
     * Returns average value in each dimensions for vectors. Reasonable default option.
     */
    Pooling AVG = vectors -> {
        double[] sum = Pooling.SUM.apply(vectors);
        for (int i = 0; i < sum.length; i++) {
            sum[i] /= vectors.length;
        }
        return sum;
    };

    /**
     * Returns sum of values in each dimensions for vectors.
     * Might be problematic if there is a mix of long and short documents.
     */
    Pooling SUM = vectors -> {
        int dimensions = vectors[0].length;
        double[] sum = new double[dimensions];
        for (double[] vector : vectors) {
            for (int d = 0; d < dimensions; d++) {
                sum[d] += vector[d];
            }
        }
        return sum;
    };

    /**
     * Returns maximum value in each dimensions for vectors.
     */
    Pooling MAX = vectors -> {
        int dimensions = vectors[0].length;
        double[] maxFeatures = new double[dimensions];
        for (int dimension = 0; dimension < dimensions; dimension++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < vectors.length; i++) {
                double feature = maxFeatures[dimension];
                max = Math.max(feature, max);
                maxFeatures[dimension] = max;
            }
        }
        return maxFeatures;
    };

    /**
     * Returns minimum value in each dimensions for vectors.
     */
    Pooling MIN = vectors -> {
        int dimensions = vectors[0].length;
        double[] minFeatures = new double[dimensions];
        for (int dimension = 0; dimension < dimensions; dimension++) {
            double min = Double.POSITIVE_INFINITY;
            for (int i = 0; i < vectors.length; i++) {
                double feature = minFeatures[dimension];
                min = Math.min(feature, min);
                minFeatures[dimension] = min;
            }
        }
        return minFeatures;
    };

}
