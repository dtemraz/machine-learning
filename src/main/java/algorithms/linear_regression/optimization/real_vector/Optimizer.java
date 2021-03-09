package algorithms.linear_regression.optimization.real_vector;

/**
 * This interface lets user configure and submit parameter optimization techniques to {@link algorithms.linear_regression.LogisticRegression}
 * Currently, only variations of gradient descent are supported.
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface Optimizer {

    /**
     * This method optimizes <em>coefficients</em> for classification of <em>trainingSet</em> using one of the
     * <strong>gradient descent</strong> techniques.
     *
     * @param trainingSet each row represent a single learning sample whose label is found in a matching column in <em>expected</em>
     * @param expected class labels associated with <em>trainingSet</em>
     * @param coefficients to optimize with <strong>bias coefficient</strong> in the <em>last</em> position
     */
    void optimize(double[][] trainingSet, double[] expected, double[] coefficients);

}
