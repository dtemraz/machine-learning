package algorithms.model;

/**
 * This interface defines generic model which can classify words and provide probability for the prediction.
 *
 * @author dtemraz
 */
public interface TextModelWithProbability extends TextModel {

    /**
     * Returns predicted class and probability for the classification together with probabilities per each class.
     * <p>
     * The ordering of probabilities in a probability vector will be <strong>same as the ordering of keys in
     * learning set </strong> used to train model.
     * </p>
     *
     * @param words to classify
     * @return predicted class and probability for <em>words</em>
     */
    ClassificationResult classifyWithProb(String[] words);
}