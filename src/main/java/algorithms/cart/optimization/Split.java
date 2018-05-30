package algorithms.cart.optimization;

import java.util.List;

/**
 * This class is a model for split in decision tree.
 *
 * @author dtemraz
 */
public class Split {
    int index; // index of attribute for which split was made
    double value; // value of attribute for which split was made
    double score; // score of cost function
    List<double[]> bellow; // samples with indexed attribute value bellow splitting value
    List<double[]> above; // samples with indexed attribute value above splitting value

    public int getIndex() {
        return index;
    }

    public double getValue() {
        return value;
    }

    public double getScore() {
        return score;
    }

    public List<double[]> getBellow() {
        return bellow;
    }

    public List<double[]> getAbove() {
        return above;
    }
}
