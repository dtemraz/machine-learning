package algorithms.neural_net;

import utilities.math.Vector;

/**
 * This class computes stable SoftMax function which is resistant to <em>underflow</em> and <em>overflow</em> by subtracting max component
 * from all other components in a vector before computing SoftMax.
 *
 * @author dtemraz
 */
public class StableSoftMaxActivation {

    /**
     * Computes numerically stable SoftMax function for <em>input</em> vector. This implementation is more robust against <em>underflow</em> and <em>overflow</em>
     * than the standard {@link SoftMaxActivation#apply(double[])}.
     *
     * @param input vector for which to compute SoftMax function
     * @return SoftMax function for <em>input</em> vector
     */
    public static double[] apply(double[] input) {
        double max = Vector.max(input);
        for (int component = 0; component < input.length; component++) {
            input[component] -= max;
        }
        return SoftMaxActivation.apply(input);
    }

}
