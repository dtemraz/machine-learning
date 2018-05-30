package structures.text;

/**
 * @author dtemraz
 */
public class Term {

    private final int id;
    double idf;

    Term(int id, double idf) {
        this.id = id;
        this.idf = idf;
    }

    public int getId() {
        return id;
    }

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
