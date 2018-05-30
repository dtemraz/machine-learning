package algorithms.cart.optimization;

import algorithms.cart.ClassificationTree;

import java.util.HashMap;
import java.util.List;

/**
 * This interface defines cost function for {@link ClassificationTree}. The cost function is used to evaluate quality of the
 * split in the tree.
 *
 * @author dtemraz
 */
public interface CostFunction {

    /**
     * Calculates cost function value for this <em>dataSet</em> of vectors. The method assumes that class id is the last vector
     * component.
     *
     * @param dataSet for which to calculate cost function value
     * @return cost function value for this <em>dataSet</em> of vectors
     */
    double apply(List<double[]> dataSet);

    /* implementation of gini index cost function */

    CostFunction GINI_INDEX = group -> {
        if (group.size() == 0) {
            return 0;
        }

        // map of class labels and their frequencies
        HashMap<Double, Integer> classFrequencies = new HashMap<>();
        // all rows have same length and class id in same position, so take first(any) to find class index in array
        int classIndex = group.get(0).length - 1;
        group.stream().map(row -> row[classIndex]).forEach(row -> classFrequencies.merge(row, 1, (old, n) -> old + n));

        double squaredFractionsSum = classFrequencies.values().stream().map(fraction -> {
            double ratio = fraction / group.size();
            return ratio * ratio;
        }).reduce(Double::sum).get();

        return 1 - squaredFractionsSum;
    };

}
