package algorithms;

import algorithms.ensemble.model.Model;
import algorithms.ensemble.model.TextModel;

import java.io.Serializable;

/**
 * This class is a thin decorator for {@link TextModel} and {@link Model} which lets user receive concrete class labels 0 or 1 as outputs
 * rather than probability value.
 * The class is decided based on the {@link #threshold}, probabilities above threshold are rounded up to 1 and those bellow
 * are rounded down to 0.
 * <p>
 * The class may be used to change balance between false positives and false negative for models which return probability.
 * </p>
 *
 * @author dtemraz
 */
public class WithThreshold implements TextModel, Model, Serializable {

    private static final long serialVersionUID = 1L;

    private final double threshold; // outputs above are rounded up to 1, those bellow are rounded down to 0

    /* user may pass only one of these to factory method */

    private final TextModel textModel; // trained instance of text model to decorate with threshold
    private final Model model; // trained instance of model to decorate with threshold

    /**
     * Decorates <em>textModel</em>'s output with rounding to nearest integer above or bellow <em>threshold</em>.
     * This may be useful for models which output probability but we only need target class, or to <strong>change</strong> the balance
     * between false positives and false negatives.
     *
     * @param textModel to decorate with threshold
     * @param threshold for which to round the output above or bellow
     * @return <em>textModel</em> decorated to return value rounded to the nearest above or bellow threshold
     */
    public static TextModel textModel(TextModel textModel, double threshold) {
        return new WithThreshold(textModel, null, threshold);
    }

    /**
     * Decorates <em>model</em>'s output with rounding to nearest integer above or bellow <em>threshold</em>.
     * This may be useful for models which output probability but we only need target class, or to <strong>change</strong> the balance
     * between false positives and false negatives.
     *
     * @param model to decorate with threshold
     * @param threshold for which to round the output above or bellow
     * @return <em>model</em> decorated to return value rounded to the nearest above or bellow threshold
     */
    public static Model model(Model model, double threshold) {
        return new WithThreshold(null, model, threshold);
    }

    private WithThreshold(TextModel textModel, Model model, double threshold) {
        if (threshold <= 0 || threshold >= 1) {
            throw new IllegalArgumentException("threshold must be between 0 and 1, exclusive but was: " + threshold);
        }
        this.textModel = textModel;
        this.model = model;
        this.threshold = threshold;
    }

    @Override
    public double predict(double[] data) {
        return model.predict(data) > threshold ? 1D : 0D;
    }

    @Override
    public double classify(String[] words) {
        return textModel.classify(words) > threshold ? 1D : 0D;
    }

}
