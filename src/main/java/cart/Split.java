package cart;

import java.util.List;

/**
 * This class is a model for split in decision tree.
 *
 * @author dtemraz
 */
class Split {
    int index; // index of attribute for which split was made
    double value; // value of attribute for which split was made
    double score; // score of cost function
    List<double[]> bellow; // samples with indexed attribute value bellow splitting value
    List<double[]> above; // samples with indexed attribute value above splitting value

    Split(int index, double value, double score, List<double[]> bellow, List<double[]> above) {
        this.index = index;
        this.value = value;
        this.score = score;
        this.bellow = bellow;
        this.above = above;
    }

}
