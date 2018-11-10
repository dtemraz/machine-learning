package algorithms.model;

import java.util.List;

/**
 * This interface defines parametric variant of {@link java.util.function.Supplier} for a {@link Model}.
 * The model should be fully initialized and trained for classification or regression given the data set
 *
 * @author dtemraz
 */
public interface ModelSupplier {
    /**
     * Create and trains model instance given <em>dataSet</em> samples
     *
     * @param dataSet samples for which to create and train model instance
     * @return model trained with <em>dataSet</em> samples
     */
    Model get(List<double[]> dataSet);

}
