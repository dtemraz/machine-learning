package algorithms.linear_regression.optimization.real_vector;

/**
 * This interface defines function that should calculate output from the given input sample and theta.
 * The {@link java.util.function.Function} interface is not used since it cannot be parametrized with primitive arrays.
 *
 * @author dtemraz
 */
public interface Predictor {

    /**
     * Returns output for this <em>input</em> and <em>theta</em>
     *
     * @param input sample for which to calculate output
     * @param coefficients to use with input
     * @return output of predictor operation over <em>input</em> and <em>theta</em>
     */
    double apply(double[] input, double[] coefficients);
}