package structures.text;

/**
 * This class extends {@link Term} with TF-IDF value.
 * The term's idf is a global value and can be precomputed in advance, tf is a local value which depends on the document.
 *
 * @author dtemraz
 */
public class TF_IDF_Term extends Term {

    private final double tfIdf;

    TF_IDF_Term(int id, double idf, double tf) {
        super(id, idf);
        tfIdf = tf * idf;
    }

    /**
     * Returns tf-idf value associated with this term.
     *
     * @return tf-idf associated with this term
     */
    public double getTfIdf() {
        return tfIdf;
    }

}
