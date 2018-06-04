package algorithms.linear_regression.optimization.text;

import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Term;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Vocabulary;

import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * This class is a wrapper for TF-IDF values of words in messages associated with class with the {@link #classId}.
 * The class contain method {@link #extractSamples(Map, Vocabulary)} to convert known data, given the vocabulary, into
 * TF-IDF metric space.
 * <p>
 * Performance notes:
 * <ul>
 *  <li>This operation should only be executed once before gradient descent starts</li>
 *  <li>TF_IDF_Terms are stored in an array since profiler has shown that {@link HashSet#iterator()} is one of bottlenecks</li>
 * </ul>
 * </p>
 *
 * @author dtemraz
 */
@RequiredArgsConstructor
class TextSample {

    final double classId; // class id associated with terms
    final TF_IDF_Term[] terms; // tf-idf term for each word in a message

    /**
     * Extracts TF-IDF value for each word in each message from <em>data</em>.
     *
     * @param data messages broken into words per class
     * @param vocabulary of all possible words
     * @return TF-IDF metric for each word from <em>data</em>
     */
    static TextSample[] extractSamples(Map<Double, List<String[]>> data, Vocabulary vocabulary) {
        // each message has a single TextSample instance associated
        TextSample[] textSamples = new TextSample[data.values().stream().map(List::size).reduce(Integer::sum).get()];
        int sample = 0;
        for (Map.Entry<Double, List<String[]>> entry : data.entrySet()) {
            double expectedClass = entry.getKey();
            List<String[]> samples = entry.getValue();
            // calculate tf-idf for each word in each sample
            for (String[] text : samples) {
                textSamples[sample++] = new TextSample(expectedClass, TF_IDF_Vectorizer.tfIdf(text, vocabulary));
            }
        }
        return textSamples;
    }
}
