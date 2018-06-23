package utilities;

import java.util.Random;

/**
 * This class offers utility methods for vector operations and by vector it's meant array. However, it is useful to think
 * in terms of vectors within machine learning domain.
 * 
 * <p>
 * The method {@link #dotProduct(double[], double[])} calculates dot product of two vectors,
 * and {@link #copyWithFirst(double[], double)} copies the content of given array into new array and injects the 0th value
 * in the new array equal to second argument.
 * </p>
 *
 * @author dtemraz
 */
public class Vector {
    
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
     * Multiplies vector <em>v</em> by magnitude <em>x</em> and returns new vector as a result.
     *
     * @param v vector to multiply by magnitude <em>x</em>
     * @param x magnitude to multiply vector <em>v</em>
     * @return vector <em>v</em> multiplied by magnitude <em>x</em>.
     */
    public static double[] multiply(double[] v, double x) {
        double[] multiplied = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            multiplied[i] = v[i] * x;
        }
        return multiplied;
        // the line bellow completely kills performance, keeping it to remind myself that (small) streams are currently BUG in java
        // granted, iterators would also be slower, but not this much slower
        // return Arrays.stream(v).map(component -> component * x).toArray();
    }

    /**
     * Perform an in-place summation of components of these two vectors, saving result in vector <em>v1</em>.
     *
     * @param v1 vector to mutate and sum with <em>v2</em>
     * @param v2 vector to sum with <em>v1</em>
     * @throws IllegalArgumentException if vectors are of different dimension
     */
    public static void mergeSum(double[] v1, double[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("invalid sample, input and v2 vectors are of different dimensions");
        }
        for (int component = 0; component < v1.length; component++) {
            v1[component] = v1[component] + v2[component];
        }
    }

    /**
     * Copies the content of vector into new vector and sets the first as 0th value in the new vector.
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
     * Returns array of a specified size filled with random doubles of a value between -0.5 and 0.5.
     * 
     * @param size number of elements to generate
     * @return random array with values between -0.5 and 0.5
     */
    public static double[] randomArray(int size) {
        double[] random = new double[size];
        for (int i = 0; i < random.length; i++) {
            random[i] += Math.random() - 0.5;
        }
        return random;
    }

    /**
     * Merges vectors <em>v1</em> and <em>v2</em> into a new vector containing both.
     *
     * @param v1 vector to merge with v2
     * @param v2 vector to merge with v1
     * @return merged vectors v1 and v2
     */
    public static double[] merge(double[] v1, double[] v2) {
        double[] merged = new double[v1.length + v2.length];
        System.arraycopy(v1, 0, merged, 0, v1.length);
        System.arraycopy(v2, 0, merged, v1.length, v2.length);
        return merged;

    }

    /**
     * Implements of Knuth's shuffle which is guaranteed to be uniform and run in linear time.
     *
     * @param items to shuffle
     */
    public static void shuffle(Object[] items) {
        Random random = new Random();
        for (int current = items.length; current > 1; current--) {
            exchange(items, current - 1, random.nextInt(current));
        }
    }

    // exchange item at position i with item at position j in items array
    private static void exchange(Object[] items, int i, int j) {
        Object tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }

}