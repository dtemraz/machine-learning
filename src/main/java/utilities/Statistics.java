package utilities;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * This class is a utility class for common statistics operations. The class offers method to calculate chiSquare of a matrix
 * {@link #chiSquare(int[][])}, data mean {@link #mean(double[])}, variance {@link #var(double[])}, covariance
 * {@link #cov(double[], double[])} and standard deviation {@link #stDev(double[])}.
 * 
 * @author dtemraz
 */
public class Statistics {

    /**
     * Calculates Chi-squared value from <em>observation</em> matrix. This test is used to determine, together with p-value
     * if there is a a significant difference between the expected frequencies and the observed frequencies.
     *
     * @param observations for which to calculate chi-squared value
     * @return chi-squared value from observations
     */
    public double chiSquare(int[][] observations) {
        return ChiSquare.calculate(observations);
    }

    /**
     * Calculates arithmetic mean(average) for <em>data</em>.
     *
     * @param data for which to calculate average value
     * @return average value of data
     */
    public static double mean(double[] data) {
        return Arrays.stream(data).reduce(Double::sum).getAsDouble() / data.length;
    }

    /**
     * Calculates variance for <em>data</em>.
     *
     * @param data for which to calculate variance
     * @return variance of data
     */
    public static double var(double[] data) {
        double stDev = stDev(data);
        return stDev * stDev;
    }

    /**
     * Calculates covariance between variables <em>x</em> and <em>y</em>.
     *
     * @param x for which to calculate covariance with <em>y</em>
     * @param y for which to calculate covariance with <em>x</em>
     * @return covariance between <em>x</em> and <em>y</em>
     */
    public static double cov(double[] x, double[] y) {
        /* covariance measures how two variables change together */
        double meanX = mean(x);
        double meanY = mean(y);
        double cov = 0;
        int samples = x.length;
        for (int i = 0; i < samples; i++) {
            cov += (x[i] - meanX) * (y[i] - meanY);
        }
        int degreesOfFreedom = samples - 1;
        return cov / degreesOfFreedom;
    }

    /**
     * Calculates standard deviation for <em>data</em>.
     *
     * @param data for which to calculate standard deviation
     * @return standard deviation of data
     */
    public static double stDev(double[] data) {
        double mean = mean(data);
        double sum = 0;
        for (double x : data) {
            double delta = x - mean;
            sum += delta * delta;
        }
        int degreesOfFreedom = data.length - 1;
        return Math.sqrt(sum / degreesOfFreedom);
    }

    /**
     * Returns the most frequent element in <em>data</em>. Method runs in O(n) time and O(n) space complexity.
     *
     * @param data for which to find mode(most frequent element)
     * @return the most frequent element in <em>data</em>
     */
    public static double mode(double[] data) {
        Map<Double, Integer> frequencies = new HashMap<>();
        Integer maxFrequency = Integer.MIN_VALUE;
        Double mode = Double.NaN;
        for (double sample : data) {
            Integer frequency = frequencies.merge(sample, 1, (old, n) -> old + n);
            if (frequency > maxFrequency) {
                mode = sample;
                maxFrequency = frequency;
            }
        }
        return mode;
    }

}