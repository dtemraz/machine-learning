package ensemble.model.processor;

import ensemble.model.Model;

import java.util.List;

/**
 * This interface defines method to evaluate ensemble of {@link Model} given the data. Expected implementations are
 * sequential and parallel execution.
 *
 * @author dtemraz
 */
public interface EnsembleModelProcessor {

    /**
     * Returns prediction for each model in ensemble given <em>data</em> sample.
     *
     * @param ensemble models that should make prediction
     * @param data for which to make prediction
     * @return prediction for each model in ensemble given <em>data</em> sample
     */
    double[] predictions(List<Model> ensemble, double[] data);
}
