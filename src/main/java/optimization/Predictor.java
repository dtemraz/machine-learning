package optimization;

/**
 * This interface defines function that should calculate output from the given input sample and coefficients.
 * The {@link java.util.function.Function} interface is not used since it cannot be parametrized with primitive arrays.
 *
 * @author dtemraz
 */
public interface Predictor {

    /**
     * Returns output for this <em>input</em> and <em>coefficients</em>
     *
     * @param input sample for which to calculate output
     * @param coefficients to use with input
     * @return output of predictor operation over <em>input</em> and <em>coefficients</em>
     */
    double apply(double[] input, double[] coefficients);
}