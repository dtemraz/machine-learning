package structures.text;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * This class indexes unique words seen in learning set and associates these words with their inverse document frequency (IDF) .
 * Each word defines unique component in learning vector and therefore size of vector will be equal to number of unique words.
 * Component index and IDF values are encapsulated with a {@link Term} objects.
 * <p>
 * The vocabulary is useful for gradient descent optimizers which may lookup IDF values and mentioned vector component indexes
 * only for words <strong>present</strong>in a message. This ensures that dot product calculation can be done fast since other components
 * can be ignored as their value will be 0.
 * </p>
 *
 * Note that there are many ways to calculate IDF, and this implementation assumes the following:
 * <strong>IDF = 1 + Math.log(documents / documents_containing_term).</strong>
 * <p>
 * This smoothing will give a bit stronger relevance to words which are present in all documents compared to <strong>unseen</strong> words.
 * </p>
 *
 * @author dtemraz
 */
public class Vocabulary implements Serializable {

    private static final long serialVersionUID = 1L;

    private final HashMap<String, Term> terms = new HashMap<>(); // unique word associated with matching index in learning vector and IDF

    public Vocabulary(List<String[]> documents) {
        storeTerms(documents); // generate unique index for each word in a learning vector abstraction
        computeIdf(documents.size()); // calculate IDF for each term
    }

    /**
     * Returns {@link Term} with component index and IDF value for this word, or null if this is unseen word.
     *
     * @param word for which to return component index and IDF
     * @return {@link Term} with component index and IDF value for this word, or <em>null</em>if this is unseen word
     */
    public Term get(String word) {
        return terms.get(word);
    }

    /**
     * Returns all {@link Term} instances in this vocabulary.
     *
     * @return all {@link Term} instances in this vocabulary
     */
    public Collection<Term> getTerms() {
        return terms.values();
    }

    /**
     * Returns number of terms in this vocabulary.
     *
     * @return number of terms in this vocabulary.
     */
    public int size() {
        return terms.size();
    }

    // saves unique words and their total occurrences in documents
    private void storeTerms(List<String[]> documents) {
        // eliminate duplicate words from a message since IDF considers only word presence in a document
        List<Set<String>> setDocuments = documents.stream().map(s -> new HashSet<>(Arrays.asList(s))).collect(Collectors.toList());
        setDocuments.stream().flatMap(Set::stream)
                .forEach(word -> terms.merge(word, new Term(size(), 1), (old, n) -> {
                    old.idf++;
                    return old;
                }));
    }

    // there are many ways to compute idf...
    // https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity
    private void computeIdf(int documents) {
        // give stronger relevance to words which are present in all documents compared to unseen words with smoothing
        terms.values().forEach(term -> term.idf = 1 + Math.log(documents / term.idf));
    }

}