package structures.text;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author dtemraz
 */
public class TF_IDF_Vectorizer {

    private static final int NO_TARGET = Integer.MIN_VALUE; // there is no target class in operative mode

    public static Map<Double, List<double[]>> vectorize(Map<Double, List<String[]>> samples, Vocabulary vocabulary) {
        HashMap<Double, List<double[]>> vectorizedSamples = new HashMap<>();
        for (Map.Entry<Double, List<String[]>> entry : samples.entrySet()) {
            Double expectedClass = entry.getKey();
            List<double[]> vectorizedText = entry.getValue().stream().map(w -> vectorize(w, vocabulary)).collect(Collectors.toList());
            vectorizedSamples.put(expectedClass, vectorizedText);
        }
        return vectorizedSamples;
    }

    /**
     * Returns vectorized <em>words</em> given the <em>vocabulary</em>. Words not present in the vocabulary are ignored.
     *
     * @param words
     * @param vocabulary
     * @return
     */
    public static double[] vectorize(String[] words, Vocabulary vocabulary) {
        return vectorize(words, vocabulary, new double[vocabulary.size()], NO_TARGET);
    }

    public static double[] vectorize(String[] words, Vocabulary vocabulary, double targetClass) {
        return vectorize(words, vocabulary, new double[vocabulary.size() + 1], targetClass);
    }

    // vector containing TF-IDF for each word in the vocabulary is returned
    private static double[] vectorize(String[] words, Vocabulary vocabulary, double[] vectorized, double targetClass) {
        Map<Term, Double> termFrequencies = vocabulary.termFrequencies(words);
        HashSet<Term> seen = new HashSet<>();

        for (String w : words) {
            Term term = vocabulary.get(w);
            // term cannot be null in learning phase, however new messages will likely have unseen words
            if (term == null || seen.contains(term)) {
                continue;
            }
            // set value of vector component per each term to it's TF-IDF value
            vectorized[term.getId()] += (termFrequencies.get(term) * term.getIdf());
            seen.add(term);
        }

        if (targetClass != NO_TARGET) {
            vectorized[vectorized.length - 1] = targetClass;
        }
        return vectorized;
    }


}