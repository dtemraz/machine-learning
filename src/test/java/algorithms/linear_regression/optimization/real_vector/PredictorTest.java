package algorithms.linear_regression.optimization.real_vector;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PredictorTest {

    private static final double delta = 0.0001;
    private static final double[] sample = new double[]{2, 3};

    // in all tests expected values are computed as sigmoid(W.t * X + bias)

    @Test
    public void zeroBiasPrediction() {
        // given
        double[] coefficients = new double[]{0.1, 0.2, 0};
        // when
        double output = Predictor.SIGMOID.apply(sample, coefficients);
        // then
        assertEquals(0.68997, output, delta);
    }

    @Test
    public void nonZeroBiasPrediction() {
        // given
        double[] coefficients = new double[]{0.1, 0.2, 1};
        // when
        double output = Predictor.SIGMOID.apply(sample, coefficients);
        // then
        assertEquals(0.85814, output, delta);
    }

    @Test
    public void allZeroCoefficientsMidValue() {
        // given
        double[] coefficients = new double[]{0, 0, 0};
        // when
        double output = Predictor.SIGMOID.apply(sample, coefficients);
        // then
        assertEquals(0.5, output, delta);
    }

    @Test
    public void largePositiveCoefficientsConvergeToOne() {
        // given
        double[] coefficients = new double[]{50, 25, 1};
        // when
        double output = Predictor.SIGMOID.apply(sample, coefficients);
        // then
        assertEquals(1, output, delta); // sigmoid should converge to 1 at x = 10
    }

    @Test
    public void largeNegativeCoefficientsConvergeToZero() {
        // given
        double[] coefficients = new double[]{-50, -25, -1};
        // when
        double output = Predictor.SIGMOID.apply(sample, coefficients);
        // then
        assertEquals(0, output, delta); // sigmoid should converge to 0 at x = - 10
    }

}
