package supervised.learning;

/**
 * This class offers utility methods for vector operations. The method {@link #dotProduct(double[], double[])} calculates
 * dot product of two vectors, and {@link #copyWithFirst(double[], double)} copies the content of given array into new array
 * and injects the 0th value in the new array equal to second argument.
 * 
 * @author dtemraz
 */
public class VectorUtils {
    
    /**
     * Returns dot product of v1 and v2 vector.
     * 
     * @param v1 vector for which to calculate dot product with v2
     * @param v2 vector for which to calculate dot product with v1
     * @return dot product of v1 and v2 vector
     * @throws IllegalArgumentException if vectors are of different dimension
     */
    public static double dotProduct(double[] v1, double[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("invalid sample, input and v2 vectors are of different dimensions");
        }
        
        double sum = 0;
        for (int component = 0; component < v1.length; component++) {
            sum += v1[component] * v2[component];
        }
        return sum;
    }
    
    /**
     * Copies the content of vector into new vector and sets the first as 0th
     * value in the new vector.
     * 
     * @param vector to copy
     * @param first element to inject into new array at 0th position
     * @return extend copy of vector with first set as 0th value
     */
    public static double[] copyWithFirst(double[] vector, double first) {
        double[] copy = new double[vector.length + 1];
        copy[0] = first;
        System.arraycopy(vector, 0, copy, 1, vector.length);
        return copy;
    }
    
    /**
     * Returns array of a specified size filled with random
     * doubles of a value between -0.5 and 0.5.
     * 
     * @param size
     * @return random array with values between -0.5 and 0.5
     */
    public static double[] randomArray(int size) {
        double[] random = new double[size];
        for (int i = 0; i < random.length; i++) {
            random[i] += Math.random() - 0.5;
        }
        return random;
    }
}