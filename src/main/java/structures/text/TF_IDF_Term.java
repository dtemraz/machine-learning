package structures.text;

/**
 * @author dtemraz
 */
public class TF_IDF_Term extends Term {

    private final double termFrequency;
    private final double tfIdf;

    TF_IDF_Term(int id, double idf, double tf) {
        super(id, idf);
        termFrequency = tf;
        tfIdf = tf * idf;
    }

    public double getTfIdf() {
        return tfIdf;
    }

    public double getTermFrequency() {
        return termFrequency;
    }

}
