package utilities;

import java.util.HashMap;

/**
 * TODO
 * @author dtemraz
 */
class ChiSquare {

    static double calculate(int[][] observationTable) {
        int total = Matrix.sumElements(observationTable);
        double chiSquared = 0;
        HashMap<Integer, Integer> rowSums = Matrix.rowTotals(observationTable);
        HashMap<Integer, Integer> colSums = Matrix.colTotals(observationTable);
        for (int row = 0; row < observationTable.length; row++) {
            for (int col = 0; col < observationTable[0].length; col++) {
                chiSquared += chiSquaredPart(observationTable, rowSums.get(row), colSums.get(col), row, col, total);
            }
        }
        return chiSquared;
    }
    
    private static double chiSquaredPart(int[][] observations, int rowTotal, int colTotal, int row, int col, int total) {
        double estimated = (rowTotal * colTotal) / total;
        double delta = (observations[row][col] - estimated);
        return (delta * delta) / estimated;
    }

}
