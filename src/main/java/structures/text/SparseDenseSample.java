package structures.text;

import java.util.List;

/**
 * This class represents learning sample which contains features divided into <em>sparse</em> and <em>dense</em>(usually latent features)
 * with associated class id.
 * <p>
 * This could be relevant in stacked generalization where one of the approaches is to train <strong>level-0</strong> models with sparse features
 * and to train <strong>combiner</strong> with level-0 outputs + dense features.
 * </p>
 *
 * @author dtemraz
 */
public class SparseDenseSample {

    private final double classId; // target class for this learning sample
    private final List<String[]> sparse; // features such as individual words
    private final List<String[]> dense; // latent features, such as text length

    // cannot use map since it will remove duplicates
    public SparseDenseSample(double classId, List<String[]> sparse, List<String[]> dense) {
        this.classId = classId;
        this.sparse = sparse;
        this.dense = dense;
    }

    /**
     * Returns list of sparse features associated with this sample.
     *
     * @return list of sparse features associated with this sample
     */
    public List<String[]> getSparse() {
        return sparse;
    }

    /**
     * Returns list of dense features associated with this sample.
     *
     * @return list of dense features associated with this sample
     */
    public List<String[]> getDense() {
        return dense;
    }

    /**
     * Returns target class id for this sample.
     *
     * @return target class id for this sample
     */
    public double getClassId() {
        return classId;
    }

}
