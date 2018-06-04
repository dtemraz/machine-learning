package structures.text;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This class offers methods to calculate TF-IDf values for a text given the <em>vocabulary</em> for target feature space.
 * There is a method {@link #tfIdf(String[], Vocabulary)} which returns {@link TF_IDF_Term}, an abstraction which contains
 * both tf-idf and unique word index in a {@link Vocabulary}. This method is suitable for gradient descent algorithms which
 * are tuned to work with Strings rather than doubles.
 *
 * @author dtemraz
 */
public class TF_IDF_Vectorizer {

    private static final int NO_TARGET = Integer.MIN_VALUE; // there is no target class in operative mode, only in learning

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

    /***
     * Converts map of text <em>samples</em> per class into map of TF-IDF values for each text given the <em>vocabulary</em> for target feature space.
     * Words not present in the vocabulary are ignored.
     *
     * @param samples for which to calculate TF-IDF value
     * @param vocabulary which contains all words in target feature space
     * @return map of TF-IDF values for each text given the <em>vocabulary</em> for target feature space
     */
    public static Map<Double, List<double[]>> vectorize(Map<Double, List<String[]>> samples, Vocabulary vocabulary) {
        HashMap<Double, List<double[]>> vectorizedSamples = new HashMap<>();
        // replace each word in each sample with its TF-IDF value
        for (Map.Entry<Double, List<String[]>> entry : samples.entrySet()) {
            Double expectedClass = entry.getKey();
            List<double[]> vectorizedText = entry.getValue().stream().map(w -> vectorize(w, vocabulary)).collect(Collectors.toList());
            vectorizedSamples.put(expectedClass, vectorizedText);
        }
        return vectorizedSamples;
    }

    /**
     * Returns tf-idf values for <em>words</em> given the <em>vocabulary</em>. Words not present in the vocabulary are ignored.
     *
     * @param words for which to calculate tf-idf value
     * @param vocabulary which contains all words in target feature space
     * @return tf-idf values for <em>words</em> given the <em>vocabulary</em>
     */
    public static double[] vectorize(String[] words, Vocabulary vocabulary) {
        return vectorize(words, vocabulary, new double[vocabulary.size()], NO_TARGET);
    }

    /**
     * Returns tf-idf values for <em>words</em> given the <em>vocabulary</em>, injects <em>targetClass</em> as <strong>last</strong>
     * component in the returned array.
     *
     * <p> Words not present in the vocabulary are ignored. </p>
     *
     * @param words for which to calculate tf-idf value
     * @param targetClass to inject as a last element in returned array
     * @param vocabulary which contains all words in target feature space
     * @return tf-idf values for <em>words</em> given the <em>vocabulary</em>
     */
    public static double[] vectorize(String[] words, Vocabulary vocabulary, double targetClass) {
        return vectorize(words, vocabulary, new double[vocabulary.size() + 1], targetClass);
    }

    // vector containing TF-IDF for each word in the vocabulary is returned
    private static double[] vectorize(String[] words, Vocabulary vocabulary, double[] vectorized, double targetClass) {
        for (Map.Entry<Term, Double> entry : termFrequencies(words, vocabulary).entrySet()) {
            Term term = entry.getKey();
            Double tf = entry.getValue();
            vectorized[term.getId()] += (tf * term.getIdf());
        }

        if (targetClass != NO_TARGET) {
            vectorized[vectorized.length - 1] = targetClass;
        }

        return vectorized;
    }

}
