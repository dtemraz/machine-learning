package algorithms.linear_regression;

import algorithms.ensemble.model.Model;
import optimization.Optimizer;
import utilities.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * This class implements multiple linear regression model in which user can predict value of <em>single outcome</em> variable
 * with <em>multiple explanatory variables</em>.
 * <p>
 * The regression is trained via constructor {@link MultipleLinearRegression#MultipleLinearRegression(List, Optimizer)}, and
 * afterwards user can make predictions with {@link #predict(double[])}.
 * </p>
 *
 * @author dtemraz
 */
public class MultipleLinearRegression implements Model, Serializable {

    private static final long serialVersionUID = 1L;

    private final double[] theta;
    private final double bias;

    /**
     * Constructs multiple regression model which is trained with supplied data samples and their associated expected.
     * The regression theta are fitted with <em>gradientDescent</em> instance using <strong>batch</strong> mode.
     *
     * @param trainingSet list of training samples, <strong>class id</strong>should be last element in the sample array
     * @param optimizer instance to train classifiers with gradient descent configuration
     */
    public MultipleLinearRegression(List<double[]> trainingSet, Optimizer optimizer) {
        TrainingSet trainingSamples = TrainingSet.build(trainingSet);
        double[] coefficients = Vector.randomArray(trainingSet.get(0).length + 1);
        optimizer.optimize(trainingSamples.input, trainingSamples.expected, coefficients);
        // extract bias was injected in 0th spot of theta
        bias = coefficients[0];
        theta = Arrays.copyOfRange(coefficients, 1, coefficients.length);
    }

    @Override
    public double predict(double[] data) {
        return bias + Vector.dotProduct(data, theta);
    }

}