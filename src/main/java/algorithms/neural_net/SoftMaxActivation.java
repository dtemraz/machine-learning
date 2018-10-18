package algorithms.neural_net;

import java.util.HashMap;

/**
 * @author dtemraz
 */
public class SoftMaxActivation {

    public static double[] softMax(double[] input) {
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
