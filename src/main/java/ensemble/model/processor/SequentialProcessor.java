package ensemble.model.processor;

import ensemble.model.Model;

import java.util.List;

/**
 * @author dtemraz
 */
public class SequentialProcessor {

    private SequentialProcessor() {

    }

    public static double[] predictions(List<Model> ensemble, double[] data) {
        return ensemble.stream().mapToDouble(model -> model.predict(data)).toArray();
    }
}
