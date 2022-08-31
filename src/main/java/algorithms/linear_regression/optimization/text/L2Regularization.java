package algorithms.linear_regression.optimization.text;

import com.google.common.util.concurrent.AtomicDoubleArray;
import structures.text.TF_IDF_Term;

/**
 * This class implements text classification coefficients update with L2 regularization.
 * <p>
 * Again, there is assumption of cost function and activation which generate update rule: ΔWk(n) = e(n) * η * x(n).
 * </p>
 *
 * Complete expression: {@literal Wj = Wj + (a*(Y - h(x)Xj) - lambda*a*Wj)}
 *
 * @author dtemraz
 */
public class L2Regularization {

    /**
     * Updates <em>coefficients</em> with <em>update</em> multiplied by <em>terms</em>, additionally applying regularization penalty(L2) if lambda &gt; 0.
     *
     * @param terms corresponding to message to classify
     * @param coefficients vector of all coefficients in this model
     * @param update consisting of error and learning rate product
     * @param lambda regularization constant
     * @param learningRate used to control the learning speed
     */
    public static void update(TF_IDF_Term[] terms, double[] coefficients, double update, double lambda, double learningRate) {
        for (TF_IDF_Term term : terms) {
            double regularizationPenalty = lambda * learningRate * coefficients[term.getId()];
            coefficients[term.getId()] += term.getTfIdf() * update - regularizationPenalty;
        }
    }

    public static void update(double[] features, double[] coefficients, double update, double lambda, double learningRate) {
        for (int i = 0; i < features.length; i++) {
            double regularizationPenalty = lambda * learningRate * coefficients[i];
            coefficients[i] += features[i] * update - regularizationPenalty;
        }
    }


    /**
     * Updates <em>coefficients</em> with <em>update</em> multiplied by <em>terms</em>, additionally applying regularization penalty(L2) if lambda &gt; 0.
     * <p>
     * This method is intended to be used in parallel optimizers.
     * </p>
     *
     * @param terms corresponding to message to classify
     * @param coefficients vector of all coefficients in this model
     * @param update consisting of error and learning rate product
     * @param lambda regularization constant
     * @param learningRate used to control the learning speed
     */
    public static void update(TF_IDF_Term[] terms, AtomicDoubleArray coefficients, double update, double lambda, double learningRate) {
        for (TF_IDF_Term term : terms) {
            double regularizationPenalty = lambda * learningRate * coefficients.get(term.getId());
            coefficients.addAndGet(term.getId(), term.getTfIdf() * update - regularizationPenalty);
        }
    }

}
