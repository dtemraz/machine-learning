package ensemble.model;

/**
 * This interface defines generic model which is able to make predictions, either classification or regression, given data sample.
 *
 * @author dtemraz
 */
public interface Model {

    /**
     * Return predicted value for the <em>data</em>. The predicted value could be result of classification or regression,
     * depending on the concrete model configuration.
     *
     * @param data for which to make prediction
     * @return prediction for <em>data</em>
     */
    double predict(double[] data);
}
