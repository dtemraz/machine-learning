package textEmbedding.pooling;

import utilities.math.Vector;

import java.util.Arrays;

/**
 * This class is a wrapper for {@link Pooling} implementation which applies normalisation with L2 norm on each word vector
 * before applying pooling operation.
 * Might be useful for some embeddings, although I have not really seen this works better in practice.
 */
public class PoolingWithNormalisation implements Pooling {

    private final Pooling pooling;

    public PoolingWithNormalisation(Pooling pooling) {
        this.pooling = pooling;
    }

    @Override
    public double[] apply(double[][] data) {
        return pooling.apply(normalise(data));
    }

    // normalise vectors by their l2 norm
    private static double[][] normalise(double[][] vectors) {
        double[] norms = l2Norm(vectors);
        // may decide to mutate original vectors after all
        double[][] normalisedVectors = new double[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            normalisedVectors[i] = Arrays.copyOf(vectors[i], vectors[i].length);
            for (int dimension = 0; dimension < vectors[i].length; dimension++) {
                normalisedVectors[i][dimension] /= norms[i];
            }
        }
        return normalisedVectors;
    }

    // computes l2 norm for each vector
    private static double[] l2Norm(double[][] vectors) {
        double[] norms = new double[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            norms[i] = Vector.l2Norm(vectors[i]);
        }
        return norms;
    }

}
