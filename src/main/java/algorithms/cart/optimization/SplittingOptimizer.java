package algorithms.cart.optimization;

import algorithms.cart.ClassificationTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class implements optimization method that finds best local split in decision tree according to {@link CostFunction}.
 * The class makes no assumptions about features that should be iterated, instead this is the iterable parameter of constructor
 * {@link #SplittingOptimizer(CostFunction)}.
 * Two reasonable approaches are {@link FullScanOptimizer} used in {@link algorithms.ensemble.BootstrapAggregation}
 * and {@link RandomFeaturesOptimizer} used in {@link algorithms.ensemble.RandomForest}.
 *
 * <p>
 * The {@link ClassificationTree} has access to only one method {@link #findBestSplit(List)} which should find best weightedAverageScore
 * from the supplied data set with the given cost function.
 * </p>
 *
 * Finding global optima in decision trees i NP hard problem, therefore this method which finds locally optimal split does
 * not guarantee that built tree is globally optimal.
 *
 * @author dtemraz
 */
public abstract class SplittingOptimizer {

    private final CostFunction costFunction; // to evaluate locally best splits

    /**
     * Returns instance of {@link SplittingOptimizer} that minimizes <em>costFunction</em> and evaluates features and their
     * values in order defined with <em>features</em>.
     *
     * @param costFunction to minimize with splits
     */
    public SplittingOptimizer(CostFunction costFunction) {
        this.costFunction = costFunction;
    }

    /**
     * The method returns <em>dataSet</em> {@link Split} which best minimizes {@link #costFunction}.
     * The elements in the split are divided in two groups, ones bellow splitting values and the other above the value.
     *
     * @param dataSet to split in two parts
     * @return {@link Split} which best minimizes {@link #costFunction}
     */
    public final Split findBestSplit(List<double[]> dataSet) {
        double min = Double.POSITIVE_INFINITY;
        Split minSplit = new Split();
        for (int index : getFeatures()) {
            for (double val : splittingValues(dataSet, index)) {
                // first group = items less than attribute, second group = greater or equal
                List<List<double[]>> groups = splitGroups(index, val, dataSet);
                double score = weightedAverageScore(groups);
                if (score < min) { // standard find min algorithm
                    min = score;
                    minSplit.score = score;
                    minSplit.bellow = groups.get(0);
                    minSplit.above = groups.get(1);
                    minSplit.index = index;
                    minSplit.value = val;
                }
            }
        }
        return minSplit;
    }

    /**
     * Returns iterable over feature indexes which should be considered as splitting values.
     *
     * @return iterable over feature indexes which should be considered as splitting values.
     */
    abstract protected Iterable<Integer> getFeatures();

    // splits data set in groups with rows with target attributes either bellow or above value
    private List<List<double[]>> splitGroups(int attributeIndex, double attributeValue, List<double[]> dataSet) {
        List<double[]> bellow = new ArrayList<>();
        List<double[]> above = new ArrayList<>();

        for (double[] row : dataSet) {
            if (row[attributeIndex] < attributeValue) {
                bellow.add(row);
            } else
                above.add(row);
        }

        // first item are samples bellow attribute value, second item are samples above or equal
        List<List<double[]>> groups = new ArrayList<>();
        groups.add(bellow);
        groups.add(above);
        return groups;
    }

    // calculates weighted average gini weightedAverageScore for all groups
    private double weightedAverageScore(List<List<double[]>> groups) {
        double totalSamples = groups.stream().map(List::size).reduce(Integer::sum).get();
        // calculate weighted average of a gini index for all splitGroups
        return groups.stream()
                .map(group -> costFunction.apply(group) * (group.size() / totalSamples))
                .reduce(Double::sum).get();
    }

    // returns possible splitting values for data set and indexed attribute
    private double[] splittingValues(List<double[]> dataSet, int index) {
        double[] attributeValues = attributeValues(dataSet, index);
        Arrays.sort(attributeValues);
        return splittingPoints(attributeValues);
    }

    // returns all values of an attribute defined with index from rows
    private double[] attributeValues(List<double[]> data, int index) {
        double[] attribute = new double[data.size()];
        for (int row = 0; row < data.size(); row++) {
            attribute[row] = data.get(row)[index];
        }
        return attribute;
    }

    // calculates splitting points as half a difference to nearest value
    private double[] splittingPoints(double[] data) {
        double[] splittingPoints = new double[data.length + 1];
        splittingPoints[0] = data[0] - (data[1] - data[0]) / 2;
        int last = data.length - 1;
        splittingPoints[last + 1] = data[last] + (data[last] - data[last - 1]) / 2;

        for (int val = 0; val < last; val++) {
            splittingPoints[val + 1] = data[val] + (data[val + 1] - data[val]) / 2;
        }

        return splittingPoints;
    }

}
