package structures.text;

import java.util.HashMap;
import java.util.Map;

/**
 * This class offers methods to calculate TF-IDf values for a text given the <em>vocabulary</em> for target feature space.
 * There is a method {@link #tfIdf(String[], Vocabulary)} which returns {@link TF_IDF_Term}, an abstraction which contains
 * both tf-idf and unique word index in a {@link Vocabulary}. This method is suitable for gradient descent algorithms which
 * are tuned to work with Strings rather than doubles.
 *
 * @author dtemraz
 */
public class TF_IDF_Vectorizer {

    /**
     * Returns {@link TF_IDF_Term} for each word in <em>words</em> given the <em>vocabulary</em> for target feature space.
     * Words not present in the vocabulary are ignored.
     *
     * @param words for which to compute TF-IDF
     * @param vocabulary which contains all words in target feature space
     * @return TF-IDF for each word in <em>words</em> given the <em>vocabulary</em>
     */
    public static TF_IDF_Term[] tfIdf(String[] words, Vocabulary vocabulary) {
        HashMap<Term, Double> termFrequencies = termFrequencies(words, vocabulary);
        TF_IDF_Term[] tfIdf = new TF_IDF_Term[termFrequencies.size()];
        int i = 0;
        // multiply TF with IDF for each word to get final TF-IDF value
        for (Map.Entry<Term, Double> entry : termFrequencies.entrySet()) {
            Term term = entry.getKey();
            double tf = entry.getValue();
            tfIdf[i++] = new TF_IDF_Term(term.getId(), term.getIdf(), tf);
        }
        return tfIdf;
    }

    /**
     * Returns TF values associated with {@link Term} objects for each word in <em>words</em>.
     *
     * @param words which should be converted into term frequencies
     * @param vocabulary with indexed words and their document frequencies
     * @return TF values associated with {@link Term} objects for each word in <em>words</em>
     */
    public static HashMap<Term, Double> termFrequencies(String[] words, Vocabulary vocabulary) {
        HashMap<Term, Double> termFrequencies = new HashMap<>();
        // count term occurrences for each word(term)
        for (String word : words) {
            Term term = vocabulary.get(word);
            if (term != null) {
                termFrequencies.merge(term, 1D, (old, n) -> old + n);
            }
        }
        // convert term occurrences into frequencies
        for (Term term : termFrequencies.keySet()) {
            termFrequencies.compute(term, (k, v) -> v / words.length);
        }
        return termFrequencies;
    }

}
