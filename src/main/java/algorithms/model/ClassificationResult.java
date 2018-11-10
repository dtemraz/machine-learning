package algorithms.model;

import lombok.Value;

/**
 * This class defines a simple wrapper for classification outcome which includes predicted class and probability for the classification.
 *
 * @author dtemraz
 */
@Value
public class ClassificationResult {
    private final double classId;
    private final double probability;
}
