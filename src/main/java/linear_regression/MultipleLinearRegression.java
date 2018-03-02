package linear_regression;

import optimization.GradientDescent;
import utilities.Vector;

import java.util.Arrays;

/**
 * This class implements multiple linear regression model. In other words, the class lets user apply single variable(outcome)
 * from a value of a multiple predictor variables. This model minimizes sum of squared residuals(vertical distances).
 * The user trains the model via one of the constructors and afterwards is able to make regression with {@link #predict(double[])}.
 *
 * <p>The model coefficients are fitted with stochastic gradient descent.</p>
 *
 * @author dtemraz
 */
public class MultipleLinearRegression {

    // bias is a special 0th coefficient which will always be present, regardless of data sample
    private static final double BIAS_INPUT = 1; // enables trainable constant factor(intercept) for regression

    private final RegressionFunction regression;

    /**
     * Constructs multiple regression model which is trained with supplied data samples and their associated values.
     * The regression coefficients are fitted with <em>gradientDescent</em> instance using <strong>batch</strong> mode.
     *
     * @param data from which to find regression coefficients where each row is a single data sample
     * @param values associated with data
     * @param gradientDescent instance with user specified learning rate, max epochs and stopping criteria
     */
    public MultipleLinearRegression(double[][] data, double[] values, GradientDescent gradientDescent) {
        int dimensions = data[0].length;
        // we wish to calculate bias coefficient as well with gradient descent, therefore + 1 to existing coefficients
        double[] coefficients = Vector.randomArray(dimensions + 1);
        // copy of data with bias input since we expanded coefficients to include bias
        gradientDescent.batch(copyWithBias(data), values, coefficients, Vector::dotProduct);
        // extract bias was injected in 0th spot of coefficients
        double bias = coefficients[0];
        // data passed to predict method will not contain bias input -1, therefore separate coefficients and bias for equation
        double[] coefficientsWithoutBias = Arrays.copyOfRange(coefficients, 1, coefficients.length);
        regression = vars -> bias + Vector.dotProduct(vars, coefficientsWithoutBias);
    }

    /**
     * Returns estimation with multiple linear regression for the <em>explanatory</em> variables.
     *
     * @param explanatory variables for which to calculate outcome
     * @return outcome via regression analysis for <em>explanatory</em> variables
     */
    public double predict(double[] explanatory) {
        return regression.apply(explanatory);
    }

    /**
     * Creates new copy of <em>data</em> where each array in this 2d array has additional {@link #BIAS_INPUT} in it's 0th spot.
     *
     * @param data to copy and decorate with bias
     * @return copy of data with {@link #BIAS_INPUT} in 0th spot of each array
     */
    private double[][] copyWithBias(double[][] data) {
        int rows = data.length;
        double[][] dataWithBias = new double[rows][data[0].length + 1];
        for (int row = 0; row < data.length; row++) {
            dataWithBias[row] = Vector.copyWithFirst(data[row], BIAS_INPUT);
        }
        return dataWithBias;
    }

    /**
     * This interface defines contract for single variable regression. The reason {@link java.util.function.Function}
     * interface is not used is because that one cannot be parametrized with primitive array, therefore needless
     * boxing and autoboxing is required.
     */
    @FunctionalInterface
    private interface RegressionFunction {
        double apply(double[] predictors);
    }

}