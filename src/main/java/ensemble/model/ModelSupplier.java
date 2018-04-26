package ensemble.model;

import java.util.List;

/**
 * This interface defines parametric variant of {@link java.util.function.Supplier} for a {@link Model}. It's intended use
 * is with {@link ensemble.BootstrapAggregation} which can use this interface to instantiate given model with sample of
 * data set.
 *
 * @author dtemraz
 */
public interface ModelSupplier {
    /**
     * Create model instance given <em>dataSet</em>.
     *
     * @param dataSet for which to create model instance
     * @return model with <em>dataSet</em> instance
     */
    Model get(List<double[]> dataSet);
}
