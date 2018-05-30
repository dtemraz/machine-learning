package structures.text;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author dtemraz
 */
public class Vocabulary implements Serializable {

    private static final long serialVersionUID = 1L;

    private final HashMap<String, Term> terms = new HashMap<>();

    public Vocabulary(List<String[]> documents) {
        storeTerms(documents);
        computeIdf(documents.size());
    }

    public Term get(String word) {
        return terms.get(word);
    }

    public int size() {
        return terms.size();
    }

    public HashMap<Term, Double> termFrequencies(String[] words) {
        HashMap<Term, Double> termFrequencies = new HashMap<>();
        for (String word : words) {
            Term term = terms.get(word);
            if (term != null) {
                termFrequencies.merge(term, 1D, (old, n) -> old + n);
            }
        }
        for (Term term : termFrequencies.keySet()) {
            termFrequencies.compute(term, (k, v) -> v / words.length);
        }
        return termFrequencies;
    }

    public TF_IDF_Term[] tfIdf(String[] words) {
        HashMap<Term, Double> termFrequencies = new HashMap<>();
        for (String word : words) {
            Term term = terms.get(word);
            if (term != null) {
                termFrequencies.merge(term, 1D, (old, n) -> old + n);
            }
        }
        TF_IDF_Term[] tfIdf = new TF_IDF_Term[termFrequencies.size()];
        int i = 0;
        for (Map.Entry<Term, Double> entry : termFrequencies.entrySet()) {
            Term term = entry.getKey();
            tfIdf[i++] = new TF_IDF_Term(term.getId(), term.getIdf(), entry.getValue() / words.length);
        }
        return tfIdf;
    }

    private void storeTerms(List<String[]> documents) {
        documents.stream().flatMap(Arrays::stream)
                .forEach(word -> terms.merge(word, new Term(size(), 1), (old, n) -> {
                    old.idf++;
                    return old;
                }));
    }

    private void computeIdf(int documents) {
        terms.values().forEach(term -> term.idf = 1 + Math.log(documents / term.idf));
    }

}