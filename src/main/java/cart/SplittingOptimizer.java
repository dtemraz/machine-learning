package cart;

import java.util.ArrayList;
import java.util.List;

/**
 * TODO comments
 * TODO this must be changed to use mean of neighbor values for a split rather than exact value of a splitting node
 * TODO this can be optimized to a single calculation per feature rather than iteration over all features and values per split
 *
 * @author dtemraz
 */
class SplittingOptimizer {

    private final CostFunction giniIndex;

    public SplittingOptimizer(CostFunction costFunction) {
        this.giniIndex = costFunction;
    }

    Split findBestSplit(List<double[]> dataSet) {
        double minGini = Double.POSITIVE_INFINITY;
        int attributeIndex = -1;
        double attributeValue = Double.POSITIVE_INFINITY;
        List<double[]> left = new ArrayList<>();
        List<double[]> right = new ArrayList<>();

        int classId = dataSet.get(0).length - 1;
        // for all attributes except class label
        for (int index = 0; index < classId; index++) {
            for (double[] row : dataSet) {
                // first group is less than attribute, second group is greater or equal
                List<List<double[]>> groups = splitGroups(index, row[index], dataSet);
                // calculate gini score for both groups
                double score = giniScore(groups);
                if (score < minGini) {
                    minGini = score;
                    left = groups.get(0);
                    right = groups.get(1);
                    attributeIndex = index;
                    attributeValue = row[index];
                }
            }
        }
        return new Split(attributeIndex, attributeValue, minGini, left, right);
    }


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

    // calculates weighted average gini score for all groups
    private double giniScore(List<List<double[]>> groups) {
        double totalSamples = groups.stream().map(g -> g.size()).reduce(Integer::sum).get();
        // calculate weighted average of a gini index for all splitGroups
        return groups.stream()
                .map(group -> giniIndex.apply(group) * (group.size() / totalSamples))
                .reduce(Double::sum).get();
    }

}
