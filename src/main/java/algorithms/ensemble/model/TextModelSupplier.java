package algorithms.ensemble.model;

import java.util.List;
import java.util.Map;

/**
 * This interface defines parametric variant of {@link java.util.function.Supplier} for a {@link Model}.
 * The model should be fully initialized and trained for classification or regression given the data set
 *
 * @author dtemraz
 */
public interface TextModelSupplier {
    /**
     * Create and trains model instance given <em>trainingSet</em> samples
     *
     * @param trainingSet samples for which to create and train model instance
     * @return model trained with <em>trainingSet</em> samples
     */
    TextModel get(Map<Double, List<String[]>> trainingSet);

}
