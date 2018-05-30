package algorithms.ensemble.model.processor;

import algorithms.ensemble.model.TextModel;

import java.util.List;

/**
 * This interface defines method to evaluate algorithms.ensemble of {@link TextModel} given the data. Expected implementations are
 * sequential and parallel execution.
 *
 * @see TextModel
 *
 * @author dtemraz
 */
public interface TextEnsembleModelProcessor {

    /**
     * Returns prediction for each model in algorithms.ensemble given <em>data</em> sample.
     *
     * @param ensemble models that should make prediction
     * @param data for which to make prediction
     * @return prediction for each model in algorithms.ensemble given <em>data</em> sample
     */
    double[] predictions(List<TextModel> ensemble, String[] data);
}
