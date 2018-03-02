package utilities;

import java.util.HashMap;

/**
 * TODO
 * @author dtemraz
 */
public class Matrix {

    public static double[][] transpose(double[][] matrix) {
        int height = matrix.length;
        int width = matrix[0].length;
        double[][] transposed = new double[width][height];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                transposed[col][row] = matrix[row][col];
            }
        }
        return transposed;
    }

    public static double[][] sum(double[][] a, double[][] b) {
        int height = a.length;
        int width = a[0].length;
        if (height != b.length || width != b[0].length) {
            throw new IllegalArgumentException(String.format("a and be must be of same dimension, a{%d x %d}, b{%d x %d}", height, width, b.length, b[0].length));
        }
        double[][] sum = new double[width][height];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                sum[col][row] = a[row][col] + b[row][col];
            }
        }
        return sum;
    }

    public static double[][] subtract(double[][] a, double[][] b) {
        int height = a.length;
        int width = a[0].length;
        if (height != b.length || width != b[0].length) {
            throw new IllegalArgumentException(String.format("a and be must be of same dimension, a{%d x %d}, b{%d x %d}", height, width, b.length, b[0].length));
        }
        double[][] sum = new double[width][height];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                sum[col][row] = a[row][col] - b[row][col];
            }
        }
        return sum;
    }

    public static int sumElements(int[][] matrix) {
        int sum = 0;
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                sum += matrix[row][col];
            }
        }
        return sum;
    }

    public static HashMap<Integer, Integer> rowTotals(int[][] matrix) {
        return totals(matrix, matrix.length, matrix[0].length, (o, r, c) -> o[r][c]);
    }

    public static HashMap<Integer, Integer> colTotals(int[][] matrix) {
        return totals(matrix, matrix[0].length, matrix.length, (o, r, c) -> o[c][r]);
    }

    private static HashMap<Integer, Integer> totals(int[][] matrix, int outer, int inner, IntegerValueExtractor val) {
        int total = 0;
        HashMap<Integer, Integer> totals = new HashMap<>();
        for (int row = 0; row < outer; row++) {
            for (int col = 0; col < inner; col++) {
                total += val.get(matrix, row, col);
            }
            totals.put(row, total);
            total = 0;
        }
        return totals;
    }

    public static double[] colTotals(double[][] matrix) {
        return totals(matrix, matrix[0].length, matrix.length);
    }


    private static double[] totals(double[][] matrix, int outer, int inner) {
        double[] colTotals = new double[outer];
        double total = 0;
        for (int row = 0; row < outer; row++) {
            for (int col = 0; col < inner; col++) {
                total += matrix[col][row];
            }
            colTotals[row] = total;
            total = 0;
        }
        return colTotals;
    }


    private interface IntegerValueExtractor {
        int get(int[][] matrix, int row, int col);
    }

}
