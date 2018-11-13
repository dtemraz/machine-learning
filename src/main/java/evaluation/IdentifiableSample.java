package evaluation;

import lombok.NonNull;
import lombok.Value;

/**
 * A simple wrapper class for features which includes {@link #id} for easier identification of misclassified samples.
 *
 * @author dtemraz
 */
@Value
public class IdentifiableSample {
    @NonNull
    private final String[] features; // features extracted for this sample
    private final String id; // id to have easier time of tracking a sample which generated features
}
