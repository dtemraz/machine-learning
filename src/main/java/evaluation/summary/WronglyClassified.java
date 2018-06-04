package evaluation.summary;

import lombok.Value;

/**
 * This class is a simple model for wrongly classified samples. The class ensures that expected and predicted classes are different,
 * throwing an exception otherwise.
 *
 * @author dtemraz
 */
@Value
public class WronglyClassified {

    private final double expected; // true class of a sample
    private final double predicted; // class guessed by machine learning
    private final String value; // textual representation of a sample

    /**
     * Constructs new instance, expected and predicted <strong>must</strong> be different.
     *
     * @param expected class of a sample
     * @param predicted class of a sample
     * @param value textual representation of a sample value
     * @throws IllegalArgumentException if <em>expected</em> equals <em>predicted</em>
     */
    public WronglyClassified(double expected, double predicted, String value) {
        if (expected == predicted) {
            throw new IllegalArgumentException("expected and predicted must be different but both are: " + expected);
        }
        this.expected = expected;
        this.predicted = predicted;
        this.value = value;
    }

}
