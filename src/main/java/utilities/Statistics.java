package utilities;

import java.util.Arrays;

/**
 * TODO
 * 
 * @author dtemraz
 */
public class Statistics {

    public double chiSquare(int[][] observations) {
        return ChiSquare.calculate(observations);
    }

    public static double mean(double[] data) {
        return Arrays.stream(data).reduce(Double::sum).getAsDouble() / data.length;
    }

    public static double var(double[] data) {
        double stDev = stDev(data);
        return stDev * stDev;
    }

    public static double cov(double[] x, double[] y) {
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

}