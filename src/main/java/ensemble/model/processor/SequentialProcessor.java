package ensemble.model.processor;

import ensemble.model.Model;

import java.util.List;

/**
 * This class can be used to satisfy {@link EnsembleModelProcessor#predictions(List, double[])} with sequential evaluation
 * of models.
 *
 * @author dtemraz
 */
public class SequentialProcessor {

    // no need to instantiate this class
    private SequentialProcessor() { }

    /**
     * Returns prediction for each model in ensemble in sequential mode
     *
     * @see EnsembleModelProcessor#predictions(List, double[])
     */
    public static double[] predictions(List<Model> ensemble, double[] data) {
        return ensemble.stream().mapToDouble(model -> model.predict(data)).toArray();
    }
}
