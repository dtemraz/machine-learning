package structures.text;

import java.util.List;

/**
 * @author dtemraz
 */
public class SparseDenseSample {

    private final double classId;
    private final List<String[]> sparse;
    private final List<String[]> dense;

    // cannot use map since it will remove duplicates
    public SparseDenseSample(double classId, List<String[]> sparse, List<String[]> dense) {
        this.classId = classId;
        this.sparse = sparse;
        this.dense = dense;
    }

    public List<String[]> getSparse() {
        return sparse;
    }

    public List<String[]> getDense() {
        return dense;
    }

    public double getClassId() {
        return classId;
    }

}
