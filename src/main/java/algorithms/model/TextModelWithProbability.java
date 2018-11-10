package algorithms.model;

/**
 * This interface defines generic model which can classify words and provide probability for the prediction.
 *
 * @author dtemraz
 */
public interface TextModelWithProbability extends TextModel {

    /**
     * Returns predicted class and probability for the classification
     *
     * @param words to classify
     * @return predicted class and probability for <em>words</em>
     */
    ClassificationResult classifyWithProb(String[] words);
}