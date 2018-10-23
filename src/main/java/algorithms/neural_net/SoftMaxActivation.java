package algorithms.neural_net;

import java.util.HashMap;

/**
 * This class computes SoftMax function for a given vector. There is also numerically stable version of this clas {@link StableSoftMaxActivation}
 * which is more robust against <em>underflow</em> and <em>overflow</em>.
 *
 * @author dtemraz
 */
public class SoftMaxActivation {

    /**
     * Computes SoftMax function for <em>input</em> vector.
     *
     * @param input vector for which to compute SoftMax function
     * @return SoftMax function for <em>input</em> vector
     */
    public static double[] apply(double[] input) {
        HashMap<Double, Double> exponents = new HashMap<>();
        double sum = 0;
        for (double x : input) {
            double exp = Math.pow(Math.E, x);
            sum += exp;
            exponents.put(x, exp);
        }
        double[] output = new double[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = exponents.get(input[i]) / sum;
        }
        return output;
    }


}
