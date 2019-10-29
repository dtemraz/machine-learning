package structures.text;

import java.io.Serializable;
import java.util.*;
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
 * <p>
 * Note that there are many ways to calculate IDF, and this implementation assumes the following:
 * <strong>IDF = 1 + Math.log(documents / documents_containing_term).</strong>
 * <p>
 * This smoothing will give a bit stronger relevance to words which are present in all documents compared to <strong>unseen</strong> words.
 * </p>
 *
 * @author dtemraz
 */
public class Vocabulary implements Serializable {

    public static final int NO_PRUNING = -1; // keep all words regardless how (in)frequent they might be

    private static final long serialVersionUID = 1L;

    private final HashMap<String, Term> terms; // unique word associated with matching index in learning vector and IDF

    public Vocabulary(List<String[]> documents) {
        this(documents, NO_PRUNING);
    }

    /**
     * Returns number of <em>documents</em> in which a word appears, multiple occurrences of a word in a same document are counted once.
     *
     * @param documents for which to count word presences
     * @return number of <em>documents</em> in which a word appears, multiple occurrences of a word in a same document are counted once
     */
    public static HashMap<String, Double> countWords(List<String[]> documents) {
        HashMap<String, Double> wordsCount = new HashMap<>();
        // eliminate duplicate words from a message, also IDF considers only word presence in a document
        List<Set<String>> setDocuments = documents.stream().map(s -> new HashSet<>(Arrays.asList(s))).collect(Collectors.toList());
        setDocuments.stream().flatMap(Set::stream).forEach(word -> wordsCount.merge(word, 1D, Double::sum));
        return wordsCount;
    }

    /**
     * Returns a {@link Set} of words which appear in less than <em>minCount</em> number of <em>documents</em>.
     *
     * @param documents in which to find word that appear less than <em>minCount</em> times.
     * @param minCount minimal number of documents that should contain a word
     * @return set of words which appear in less than <em>minCount</em> number of <em>documents</em>
     */
    public static Set<String> findRareWords(List<String[]> documents, int minCount) {
        return countWords(documents).entrySet().stream().filter(e -> e.getValue() < minCount).map(Map.Entry::getKey).collect(Collectors.toSet());
    }

    /**
     * Returns instance of {@link Vocabulary} from tokenized <em>documents</em> emitting all terms which appear strictly less than <em>minCount</em>.
     *
     * @param documents tokenized into words
     * @param minCount  minimal number of documents in which a word must appear, otherwise it is pruned. Multiple occurrences in same document are counted once.
     */
    public Vocabulary(List<String[]> documents, int minCount) {
        // generate unique index and IDF for each word in a learning vector abstraction
        terms = computeTermIDF(documents, minCount);
    }

    /**
     * Returns <em>true</em> if <em>word</em> is present in vocabulary, <em>false</em> otherwise.
     *
     * @param word to check if it is present in vocabulary
     * @return <em>true</em> if <em>word</em> is present in vocabulary, <em>false</em> otherwise
     */
    public boolean contains(String word) {
        return terms.containsKey(word);
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

    // returns map of unique words and their idf scores
    private HashMap<String, Term> computeTermIDF(List<String[]> documents, int minCount) {
        HashMap<String, Double> wordsCount = countWords(documents);
        int documentsCount = documents.size();
        // creates terms map without words which appear in less than minCount documents
        HashMap<String, Term> pruned = new HashMap<>();
        for (Map.Entry<String, Double> entry : wordsCount.entrySet()) {
            Double count = entry.getValue();
            if (count >= minCount) {
                // give stronger relevance to words which are present in all documents compared to unseen words
                double idf = 1 + Math.log(documentsCount / count);
                // there are many ways to compute idf..., https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity
                pruned.put(entry.getKey(), new Term(pruned.size(), idf));
            }
        }
        return pruned;
    }

}