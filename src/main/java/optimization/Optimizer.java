package optimization;

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
     * @param trainingSet where key = class and value = texts broken into words per class
     */
    void optimize(double[][] trainingSet, double[] values, double[] coefficients);
}
