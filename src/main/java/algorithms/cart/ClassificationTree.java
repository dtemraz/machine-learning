package algorithms.cart;

import algorithms.cart.optimization.CostFunction;
import algorithms.cart.optimization.FullScanOptimizer;
import algorithms.cart.optimization.Split;
import algorithms.cart.optimization.SplittingOptimizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class implements C of CART, a classification tree. This is an example of low bias, high variance classifier whose
 * performance can usually be further improved with algorithms.ensemble techniques such as {@link algorithms.ensemble.BootstrapAggregation}.
 *
 * <p>
 * The class currently cannot handle categorical input(other than binary), it could with one-hot-encoding, but not as per
 * specification of the algorithm. On the other hand, this simplifies model for features which can be represented as a
 * simple double value and this should speed up training and prediction.
 * </p>
 **
 * @author dtemraz
 */
public class ClassificationTree {

    private final int minSize; // minimal number of elements in a group after which no further splits should be made
    private final int maxDepth; // depth of the tree, should be high for algorithms.ensemble algorithms, otherwise pruning should be used
    private final SplittingOptimizer splittingOptimizer; // finds locally optimal split with respect to cost function

    private Node root; // root node of classification tree

    /**
     * Creates instance of classification tree trained with <em>dataSet</em>. The tree won't make splits which would create
     * groups with less than <em>minSize</em> or go bellow <em>maxDepth</em>.
     *
     * @param dataSet to train the tree
     * @param minSize minimal number of elements allowed in a splitting group
     * @param maxDepth maximal depth the tree is allowed to grow
     */
    public ClassificationTree(List<double[]> dataSet, int minSize, int maxDepth) {
        this(dataSet, new FullScanOptimizer(CostFunction.GINI_INDEX, dataSet.get(0).length - 1), minSize, maxDepth);
    }

    public ClassificationTree(List<double[]> dataSet, SplittingOptimizer splittingOptimizer, int minSize, int maxDepth) {
        this.minSize = minSize;
        this.maxDepth = maxDepth;
        this.splittingOptimizer = splittingOptimizer;
        buildTree(dataSet);
    }


    /* Methods bellow are used for prediction on trained tree */


    /**
     * Returns predicted class for <em>data</em>
     *
     * @param data to mostlyRepresentedClass
     * @return predicted class for <em>data</em>
     */
    public double classify(double[] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be null or empty");
        }
        return classify(root, data);
    }


    // iteratively searches for a leaf node which best captures data characteristics to make classification
    private double classify(Node node, double[] data) {
        // java does not implement tail call optimization therefore this should perform somewhat better than recursion
        while (!node.isDecisionNode()) {
            if (data[node.index] < node.value) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.predictedClass;
    }

    /* Methods bellow are used to build a tree instance from the data set */

    // recursively builds classification tree with the top-down strategy
    private void buildTree(List<double[]> dataSet) {
        root = splitNode(splittingOptimizer.findBestSplit(dataSet));
        buildTree(root, 1);
    }

    // recursively builds classification tree up to specified depth
    private void buildTree(Node node, int depth) {
        List<double[]> bellow = node.left.dataSet;
        List<double[]> above = node.right.dataSet;

        // base case - turn parent node into decision node since this split resulted with exactly same data set
        if (bellow.isEmpty() || above.isEmpty()) {
            node.toDecisionNode();
            return;
        }

        // no need to have references loiter in splitting node
        node.dataSet = null;

        // base case - reached maximum allowed depth so make leaf nodes from groups we have so far
        if (depth >= maxDepth) {
            node.left.toDecisionNode();
            node.right.toDecisionNode();
            return;
        }

        /* handle left subtree */

        // evaluate stopping conditions
        if (node.left.singleClass() || bellow.size() < minSize) {
            node.left.toDecisionNode();
        } else {
            // recursively partition left side of the tree
            // the split could also be rejected if the size was to small after split, this is not implemented
            node.left = splitNode(splittingOptimizer.findBestSplit(bellow));
            buildTree(node.left, depth + 1);
        }

        /* handle right subtree */

        // evaluate stopping conditions
        if (node.right.singleClass() || above.size() < minSize) {
            node.right.toDecisionNode();
        } else {
            // recursively partition right side of the tree
            // the split could also be rejected if the size was to small after split, this is not implemented
            node.right = splitNode(splittingOptimizer.findBestSplit(above));
            buildTree(node.right, depth + 1);
        }
    }

    // makes a split on a node to values bellow(left) and above(right) indexed value of a split
    private Node splitNode(Split split) {
        Node node = new Node();
        node.index = split.getIndex();
        node.value = split.getValue();
        // parent node should have entire data set, child nodes only respective subsets
        node.dataSet = new ArrayList<>(split.getBellow());
        node.dataSet.addAll(split.getAbove());
        node.left = new Node(split.getBellow());
        node.right = new Node(split.getAbove());
        return node;
    }

    /**
     * This class implements node in a classification tree.
     * Generally, there are two different nodes: split node and decision(leaf) node.
     * Split nodes are used to navigate search to decision nodes which make classification as most represented class
     * in their associated data set.
     *
     * Unseen data samples are compared against specific attribute values in a split node until we get to the leaf.
     */
    private static class Node {

        private static final int UNDEFINED = -1; // leaf nodes do not use index or value to navigate search

        private int index = UNDEFINED; // index of attribute by which split was made
        private double value = UNDEFINED; // value of indexed attribute by which split was made
        private List<double[]> dataSet; // leaf nodes do classification based on the most represented class in a data set
        private double predictedClass = UNDEFINED; // calculate mostly represented class only once
        private Node left; // left child
        private Node right; // right child

        private Node() { }

        private Node(List<double[]> dataSet) {
            this.dataSet = dataSet;
        }

        // remove unnecessary information for classification from the node and calculate mostly represented class from data set
        private void toDecisionNode() {
            left = null;
            right = null;
            index = UNDEFINED;
            value = UNDEFINED;
            predictedClass = mostlyRepresentedClass();
            dataSet = null;
        }

        // returns true if this node is a leaf and can be used for classification, false otherwise
        private boolean isDecisionNode() {
            return left == null && right == null;
        }

        // verifies that all rows in data set have same class
        private boolean singleClass() {
            // all samples should have class id in last position, so take any sample
            double[] any = dataSet.get(0);
            int classId = any.length - 1;
            double expectedClass = any[classId];
            // compare all other rows to first row and check if they have same class
            for (int row = 1; row < dataSet.size(); row++) {
                if (dataSet.get(row)[classId] != expectedClass) {
                    return false;
                }
            }
            return true;
        }

        // returns most represented class(highest count) in a leaf node
        private double mostlyRepresentedClass() {
            HashMap<Double, Integer> classesCount = new HashMap<>();
            // count occurrences of each class in a node data set
            dataSet.forEach(sample -> classesCount.merge(sample[sample.length - 1], 1, (old, n) -> old + n));
            // find class with most occurrences in the data set
            return classesCount.entrySet().stream()
                    .max(Comparator.comparingInt(Map.Entry::getValue))
                    .get()
                    .getKey();
        }
    }

}
