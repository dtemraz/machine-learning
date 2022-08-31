package algorithms.linear_regression.optimization.multiclass;

import structures.text.Vocabulary;

import java.util.List;
import java.util.Map;

/**
 * This interface lets user configure and run gradient descent for multi-class classification.
 */
public interface MultiClassOptimizer {

    /**
     * This method optimizes <em>coefficients</em> using one of the <strong>gradient descent</strong> techniques to achieve
     * classification of a <em>trainingSet</em>.
     *
     * @param coefficients to optimize for training set classification
     * @param trainingSet where key = class and value = texts broken into words and latent features per class
     */
    void optimize(Map<Double, List<String[]>> trainingSet, Map<Double, double[]> coefficients, Vocabulary vocabulary);

    /**
     * This method optimizes <em>coefficients</em> using one of the <strong>gradient descent</strong> techniques to achieve
     * classification of a <em>trainingSet</em>.
     *
     * @param data where key = class and value = texts broken into embedded sentences
     * @param coefficients to optimize for training set classification
     */
    void optimize(Map<Double, List<double[]>> data, Map<Double, double[]> coefficients);

}
