package algorithms.ensemble.model.processor;

import algorithms.ensemble.model.Model;
import algorithms.ensemble.model.TextModel;

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

    public static double[] predictions(List<Model> ensemble, double[] data) {
        return ensemble.stream().mapToDouble(model -> model.predict(data)).toArray();
    }

    public static double[] predictions(List<TextModel> ensemble, String[] data) {
        return ensemble.stream().mapToDouble(model -> model.classify(data)).toArray();
    }
}
