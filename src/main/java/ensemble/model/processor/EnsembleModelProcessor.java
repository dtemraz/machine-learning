package ensemble.model.processor;

import ensemble.model.Model;

import java.util.List;

/**
 * @author dtemraz
 */
public interface EnsembleModelProcessor {

    double[] predictions(List<Model> ensemble, double[] data);
}
