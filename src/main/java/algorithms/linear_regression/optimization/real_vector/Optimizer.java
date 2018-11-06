package algorithms.linear_regression.optimization.real_vector;

/**
 * This interface let's user configure and submit different gradient techniques to {@link algorithms.linear_regression.LogisticRegression}
 * algorithm.
 * The user may transparently to logistic regression define different GD approaches such as:
 * <ul>
 *  <li>stochastic</li>
 *  <li>mini batch</li>
 * </ul>
 * and their appropriate parameters.
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface Optimizer {

    /**
     * This method optimizes <em>coefficients</em> using one of the <strong>gradient descent</strong> techniques to achieve
     * classification of a <em>trainingSet</em>.
     *
     * @param coefficients to optimize for training set classification
     * @param expectedValues class labels associated with <em>trainingSet</em>
     * @param trainingSet each row represent a single learning sample whose class label is found in a matching column ni <em>expectedValues</em>
     */
    void optimize(double[][] trainingSet, double[] expectedValues, double[] coefficients);
}
