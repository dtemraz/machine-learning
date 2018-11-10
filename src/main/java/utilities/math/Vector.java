package utilities.math;

import java.util.Random;

/**
 * This class offers utility methods for vector operations and by vector it's meant array.
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
        // the line bellow completely kills performance, keeping it to remind myself that (small) streams are currently BUG in java, granted, iterators would also be slower, but not this much slower
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
     * @param first  element to inject into new array at 0th position
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
     * Merges <em>vector</em> with <em>component</em> into a new vector containing both.
     *
     * @param vector    to expand with <em>component</em>
     * @param component to add onto <em>vector</em>
     * @return new vector expanded with <em>component</em>
     */
    public static double[] merge(double[] vector, double component) {
        return merge(vector, new double[]{component});
    }

    /**
     * Returns sum of squared <em>vector</em> components.
     *
     * @param vector for which to calculate sum of squared components
     * @return sum of squared <em>vector</em> components
     */
    public static double squaredSum(double[] vector) {
        double squaredSum = 0;
        for (double component : vector) {
            squaredSum += component * component;
        }
        return squaredSum;
    }

    /**
     * Returns <em>id</em> of component with greatest value in <em>vector</em>. If there are multiple such components, id of first one in iterative scan is is returned.
     *
     * @param vector for which to find if of max value component
     * @return value of max component in <em>vector</em>
     */
    public static int  maxComponentId(double[] vector) {
        // standard find max algorithm
        double max = Double.NEGATIVE_INFINITY; // Double.MIN_VALUE is actually positive !!
        int maxIdx = Integer.MIN_VALUE;
        for (int i = 0; i < vector.length; i++) {
            double component = vector[i];
            if (component > max) {
                max = component;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Returns value of maximal component in <em>vector</em>.
     *
     * @param vector for which to max value component
     * @return value of maximal component in <em>vector</em>
     */
    public static double max(double[] vector) {
        return vector[maxComponentId(vector)];
    }

    /**
     * Implements of Knuth's shuffle which is guaranteed to be uniform, in-place and run in linear time.
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