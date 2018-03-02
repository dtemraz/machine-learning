package bayes;

import java.util.List;

/**
 * This class is a simple wrapper for a list of Strings associated with some (class)label.
 * It is particularly useful for {@link MultinomialNaiveBayes} which should associate texts with a label during training phase
 * and return the label with classification method {@link MultinomialNaiveBayes#classify(String)}.
 *
 * @author dtemraz
 */
public class TextSamples {

    private final String label; // identity or a class of these texts
    private final List<String> texts; // texts which should be associated with the label

    public TextSamples(String label, List<String> texts) {
        this.label = label;
        this.texts = texts;
    }

    /**
     * Returns label associated with texts in this instance.
     *
     * @return label associated with texts in this instance.
     */
    public String getLabel() {
        return label;
    }

    /**
     * Returns list of texts in this instance.
     *
     * @return list of texts in this instance
     */
    public List<String> getTexts() {
        return texts;
    }
}
