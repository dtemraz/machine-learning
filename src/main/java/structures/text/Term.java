package structures.text;

import java.io.Serializable;

/**
 * This class defines unique <em>id</em> for a word with associated <em>inverse document frequency(IDF)</em>. IDF can only be calculated
 * once all documents(messages) have been seen. In order to save some memory IDF initially represents term <em>occurrences</em> and is
 * incrementally updated by te {@link Vocabulary}, hence <em>package</em> access and <em>volatile</em> modifier for {@link #idf}.
 *
 * <p> Once all documents are seen the occurrences value is turned into IDF within Vocabulary building process. </p>
 *
 * @author dtemraz
 */
public class Term implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int id; // unique component id in learning vector for a word
    volatile double idf; // calculated incrementally by the Vocabulary instance, hence volatile and package access

    Term(int id, double idf) {
        this.id = id;
        this.idf = idf;
    }

    /**
     * Returns unique component id in learning vector abstraction for this term.
     *
     * @return unique component id in learning vector abstraction for this term
     */
    public int getId() {
        return id;
    }

    /**
     * Returns inverse document frequency associated with this term.
     *
     * @return inverse document frequency associated with this term
     */
    public double getIdf() {
        return idf;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Term)) return false;
        Term term = (Term) o;
        return id == term.id;
    }

    @Override
    public int hashCode() {
        return id;
    }

}
