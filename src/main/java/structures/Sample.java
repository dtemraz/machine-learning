package structures;

import lombok.Value;

/**
 * This class defines a sample which contains input values and expected target value for the input.
 *
 * @author dtemraz
 */
@Value
public class Sample {
    private final double[] values;
    private final double target;
}
