package algorithms.linear_regression.optimization.text;

import java.util.List;
import java.util.Map;

/**
 * This interface let's user configure and submit different gradient techniques to {@link algorithms.linear_regression.LogisticRegression}
 * algorithm.
 * <p>
 * The user may define different gradient descent approaches, and their parameters, such as:
 * <ul>
 *  <li>stochastic, sequential and parallel</li>
 *  <li>mini batch</li>
 * </ul>
 * </p>
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface MultiClassTextOptimizer {

    /**
     * This method optimizes <em>coefficients</em> using one of the <strong>gradient descent</strong> techniques to achieve
     * classification of a <em>trainingSet</em>.
     *
     * @param coefficients to optimize for training set classification
     * @param trainingSet where key = class and value = texts broken into words per class
     */
    void optimize(Map<Double, List<String[]>> trainingSet, Map<Double, double[]> coefficients);
}
