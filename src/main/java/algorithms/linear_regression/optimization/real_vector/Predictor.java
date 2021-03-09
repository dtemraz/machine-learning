package algorithms.linear_regression.optimization.real_vector;

import algorithms.neural_net.Activation;

/**
 * This interface defines function that should calculate estimate for the given input sample and coefficients.
 * Implementations should assume that <strong>bias</strong> is in the last position in <em>coefficients</em> array.
 *
 * @author dtemraz
 */
interface Predictor {

    /**
     * Returns prediction for <em>input</em> and <em>coefficients</em>, assuming that bias coefficient is in the last position.
     *
     * @param input sample for which to calculate prediction
     * @param coefficients for input, <strong>bias</strong> is in the <strong>last</strong> position
     * @return predicted value given <em>input</em> and <em>coefficients</em>
     */
    double apply(double[] input, double[] coefficients);

    Predictor SIGMOID = (input, coefficients) -> {
        double bias = coefficients[coefficients.length - 1];
        double dotProduct = 0;
        // input is shorter for one element than coefficients since it does not have bias signal
        for (int i = 0; i < input.length; i++) {
            dotProduct += input[i] * coefficients[i];
        }
        return Activation.SIGMOID.apply(bias + dotProduct);
    };

}