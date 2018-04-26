package cart;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

/**
 * This class implements C of CART, a classification tree. This is an example of low bias, high variance classifier whose
 * performance can usually be further improved with ensemble techniques such as {@link ensemble.BootstrapAggregation}.
 *
 * <p>
 * The class currently cannot handle categorical input, it could with one-hot-encoding, but not as per specification of
 * the algorithm. On the other hand, this simplifies model for features which can be represented as a simple double value
 * and this should speed up training and prediction considerably.
 * </p>
 **
 * @author dtemraz
 */
public class ClassificationTree {

    private final int minSize; // minimal number of elements in a group after which no further splits should be made
    private final int maxDepth; // depth of the tree, should be high for ensemble algorithms, otherwise pruning should be used
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
        this.minSize = minSize;
        this.maxDepth = maxDepth;
        splittingOptimizer = new SplittingOptimizer(CostFunction.GINI_INDEX);
        buildTree(dataSet);
    }

    /* Methods bellow are used for prediction on trained tree*/

    /**
     * Returns predicted class for <em>data</em>
     *
     * @param data to classify
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
        return classify(node);
    }

    // returns most represented class(highest count) in a leaf node
    private double classify(Node node) {
        HashMap<Double, Integer> classesCount = new HashMap<>();
        // count occurrences of each class in a node data set
        node.dataSet.stream().forEach(sample -> classesCount.merge(sample[sample.length - 1], 1, (old, n) -> old + n));
        // find class with most occurrences in the data set
        return classesCount.entrySet().stream()
                .max(Comparator.comparingInt(c -> c.getValue()))
                .get()
                .getKey();
    }

    /* Methods bellow are used to build a tree instance from the data set */


    // recursively builds classification tree with the top-down strategy
    private Node buildTree(List<double[]> dataSet) {
        root = splitNode(splittingOptimizer.findBestSplit(dataSet));
        buildTree(root, 1);
        return root;
    }

    // recursively builds classification tree up to specified depth
    private void buildTree(Node node, int depth) {
        List<double[]> bellow = node.left.dataSet;
        List<double[]> above = node.right.dataSet;

        // base case - turn parent node into isDecisionNode node since with this split we have got exactly same data set
        if (bellow.isEmpty() || above.isEmpty()) {
            node.turnIntoLeaf();
            return;
        }

        // no need to have references loiter in splitting node
        node.dataSet = null;

        // base case - we have reached maximum allowed depth so make leaf nodes from groups we have so far
        if (depth >= maxDepth) {
            // left and right node are already initialized with respective groups from the parent node so do nothing
            return;
        }

        /* no need to perform further partition if node already made perfect split, or we reached stopping criteria for size */

        // handle left subtree
        if (!node.left.singleClass() && bellow.size() > minSize) {
            // recursively partition left side of the tree
            node.left = splitNode(splittingOptimizer.findBestSplit(bellow));
            buildTree(node.left, depth + 1);
        }

        // handle right subtree
        if (!node.right.singleClass() && above.size() > minSize) {
            // recursively partition right side of the tree
            node.right = splitNode(splittingOptimizer.findBestSplit(above));
            buildTree(node.right, depth + 1);
        }
    }

    // makes a split on a node to values bellow(left) and above(right) indexed value of a split
    private Node splitNode(Split split) {
        Node node = new Node();
        node.index = split.index;
        node.value = split.value;
        // parent node should have entire data set, child nodes only respective subsets
        node.dataSet = new ArrayList<>(split.bellow);
        node.dataSet.addAll(split.above);
        node.left = new Node(split.bellow);
        node.right = new Node(split.above);
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
        private Node left; // left child
        private Node right; // right child

        private Node() { }

        private Node(List<double[]> dataSet) {
            this.dataSet = dataSet;
        }

        // remove unnecessary information for classification from the node
        private void turnIntoLeaf() {
            left = null;
            right = null;
            index = UNDEFINED;
            value = UNDEFINED;
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
    }

}
