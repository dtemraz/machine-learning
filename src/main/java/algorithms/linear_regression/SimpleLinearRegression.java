package algorithms.linear_regression;

import utilities.math.Statistics;

/**
 * This class implements a simple regression model. Once constructors finishes, class will find best fitting line according to
 * least squares criterion: Q=∑(expected(i)−calculated(i))^2 and user will be able to make single variable regression with {@link #predict(double)}.
 *
 * @author dtemraz
 */
public class SimpleLinearRegression {

    private final SimpleRegressionFunction bestFittingLine; // line equation with least prediction error

    /**
     * Constructs instance of SimpleLinearRegression which finds best fitting line for the given <em>expected</em> and <em>outputs</em>.
     *
     * @param values  from which to learn theta
     * @param outputs associated with expected
     * @throws IllegalArgumentException if <em>expected</em> and <em>outputs</em> are of different dimension
     */
    public SimpleLinearRegression(double[] values, double[] outputs) {
        if (values.length != outputs.length) {
            throw new IllegalArgumentException(String.format("vectors must be of same length, x = %d , y = %d", values.length, outputs.length));
        }
        double variance = Statistics.var(values);
        double covariance = Statistics.cov(values, outputs);
        // y = b0 - b1*x where bo = bias and b1 = slope
        double slope = covariance / variance;
        double bias = Statistics.mean(outputs) - slope * Statistics.mean(values);
        bestFittingLine = x -> bias + slope * x;
    }

    /**
     * Returns estimation with simple linear regression for this <em>explanatory</em> variable.
     *
     * @param explanatory variable for which to calculate outcome
     * @return outcome via regression analysis for <em>explanatory</em> variable
     */
    public double predict(double explanatory) {
        return bestFittingLine.apply(explanatory);
    }

    /**
     * This interface defines contract for single variable regression. The reason {@link java.util.function.Function}
     * interface is not used is because that one cannot be parametrized with primitives, therefore needless
     * boxing and autoboxing is required.
     */
    @FunctionalInterface
    private interface SimpleRegressionFunction {
        double apply(double predictor);
    }

}
