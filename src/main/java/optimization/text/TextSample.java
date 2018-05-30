package optimization.text;

import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Term;
import structures.text.Vocabulary;

import java.util.ArrayList;
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
    final TF_IDF_Term[] terms; // term for each word in each message associated with class id

    /**
     * Extracts TF-IDF value for each word in each message from <em>data</em>.
     * <p>
     * <strong>Side-effect:</strong> in order to conserve memory this method will replace all <em>data</em> values
     * with null so they are free for garbage collection. It's not ideal since client might kept references to individual
     * lists, however {@link List#clear()} is even worse since {@link java.util.ArrayList} will not invoke {@link ArrayList#trimToSize()}
     * and therefore large lists will still occupy considerable space.
     * </p>
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
                textSamples[sample++] = new TextSample(expectedClass, vocabulary.tfIdf(text));
            }
            // conserve memory, although this is somewhat ugly side-effect
            data.replace(expectedClass, samples, null);
        }
        return textSamples;
    }
}
