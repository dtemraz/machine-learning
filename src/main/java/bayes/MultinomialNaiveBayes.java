package bayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * This class implements <em>Multinomial Naive Bayes algorithm</em> with Laplace smoothing for text classification.
 * The algorithm uses Bayesian theorem: P(A|B) = (P(B|A) * P(A)) / P(B) where:
 * <ul>
 *  <li>P(A|B) conditional probability of event A, given event B </li>
 *  <li>P(B|A) conditional probability of event B, given event A</li>
 *  <li>P(A) independent probability of event A</li>
 *  <li>P(B) independent probability of event B</li>
 * </ul>
 *
 * The algorithm is referred to as naive because it assumes there are no dependencies between features which simply doesn't hold
 * for most real world scenarios. For text analysis, it assumes that distribution of a given word doesn't affect distribution(existence)
 * of other words. Regardless, even with this naive assumption algorithm works reasonably well for many real word scenarios.
 *
 * <p>
 * The algorithm is trained via constructor with samples and their associated class labels {@link TextSamples}.
 * The user may get most probable class label for a text via method {@link #classify(String)} once constructor finishes.
 * </p>
 * This algorithm is a solid choice when there are not many samples to learn from. While it may have greater asymptotic error
 * than it's counterpart linear regression model, it approaches this error faster:
 *      <p>https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf</p>
 *
 * The algorithm may still perform reasonably for large problems and given it's simplicity it is worth trying before rejecting it
 * as a solution. For large texts classification, TF-IDF metric should be considered instead of words frequency.
 * <p>
 * <strong>Corner cases:</strong>
 * <ul>
 *  <li>probability of unseen words is handled with Laplace smoothing</li>
 *  <li>numeric underflow is solved with logarithmic sum of probabilities instead of probabilities product</li>
 * </ul>
 * </p>
 *
 * @author dtemraz
 */
public class MultinomialNaiveBayes {

    private static final String WHITESPACES = "\\s+"; // covers multiple whitespaces (tabs, new lines, spaces)

    private final List<ClassParameters> classParameters; // properties of each class defined with constructor samples
    private final int possibleWords; // count of unique set of possible words from all classes

    /**
     * Constructs instance of this class trained to classify <em>samples</em> into classes defined with their labels.
     *
     * @param samples to train Multinomial Naive Bayes algorithm
     */
    public MultinomialNaiveBayes(List<TextSamples> samples) {
        if (samples == null) {
            throw new IllegalArgumentException("samples must not be null");
        }
        classParameters = train(samples);
        possibleWords = uniqueWordsCount();
        // handle unseen words since they would have probability 0 which would cause problems for logarithmic function
        classParameters.forEach(cd -> laplaceSmoothing(cd.probabilityTable));
    }

    /**
     * Returns the most likely class label for this text according to Bayesian theorem. Text will be split into words
     * with {@link #WHITESPACES} expression.
     *
     * @param text to classify into one of the labels supplied via constructor
     * @return most probable class label for this text
     */
    public String classify(String text) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("text must not be null or empty");
        }
        return classify(text.split(WHITESPACES));
    }

    /**
     * Returns the most likely class label for these words according to Bayesian theorem.
     *
     * @param words to classify into one of the labels supplied via constructor
     * @return most probable class label for this text
     */
    public String classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        return maxLikelihoodClass(words);
    }

    /*
    Learning
    -------------------------------------------------------------------------------
   */

    // trains naive bayes from the provided samples
    private List<ClassParameters> train(List<TextSamples> samples) {
        // sums sizes of texts in each sample
        int totalSamples = samples.stream().map(sp -> sp.getTexts().size()).reduce(Integer::sum).get();
        List<ClassParameters> probabilityTables = new ArrayList<>();
        samples.forEach(sample -> probabilityTables.add(getClassDistribution(sample, totalSamples)));
        return probabilityTables;
    }

    // calculates words distribution parameters for the given textSamples of a class
    private ClassParameters getClassDistribution(TextSamples textSamples, int totalSamples) {
        List<String> texts = textSamples.getTexts();
        HashMap<String, Double> frequencyTable = buildFrequencyTable(texts);
        double classProbability = (double) texts.size() / totalSamples;
        return new ClassParameters(textSamples.getLabel(), frequencyTable, classProbability, countWords(frequencyTable));
    }

    // builds frequency table of words from the provided list of texts
    private HashMap<String, Double> buildFrequencyTable(List<String> texts) {
        HashMap<String, Double> frequencyTable = new HashMap<>(); // key = word, value = count
        texts.stream().flatMap(text -> Stream.of(text.split(WHITESPACES)))
                .forEach(word -> frequencyTable.merge(word.toLowerCase(), 1D, (old, n) -> old + n));
        return frequencyTable;
    }

    // solves problem of missing words probability
    private void laplaceSmoothing(HashMap<String, Double> frequencyTable) {
        // sum of words and their frequencies given the frequency table
        int totalWordsInClass = countWords(frequencyTable);
        // add one to each frequency in the table and increase probability denominator by number of possible words
        frequencyTable.forEach((key, value) -> frequencyTable.compute(key, (k, v) -> (v + 1) / (totalWordsInClass + possibleWords)));
    }

    // returns total number of words in a frequency table
    private int countWords(HashMap<String, Double> frequencyTable) {
        Double wordsInClass = frequencyTable.entrySet().stream()
                .mapToDouble(Map.Entry::getValue)
                .reduce(Double::sum)
                .getAsDouble();
        return wordsInClass.intValue();
    }

    // returns count of unique words given all classes
    private int uniqueWordsCount() {
        HashSet<String> uniqueWords = new HashSet<>();
        classParameters.forEach(cd -> uniqueWords.addAll(cd.probabilityTable.keySet()));
        return uniqueWords.size();
    }

    /*
     Classification
     -------------------------------------------------------------------------------
     */

    // returns class with highest probability of containing words in this text
    private String maxLikelihoodClass(String[] text) {
        double max = Double.NEGATIVE_INFINITY;
        String label = "";
        // this is just a standard find max algorithm
        for (ClassParameters parameters : classParameters) {
            double probability = Math.log(parameters.classProbability) + logSumProbability(text, parameters);
            if (probability > max) {
                max = probability;
                label = parameters.label;
            }
        }
        return label;
    }

    // returns sum of logarithmic probabilities for each word, given class defined with classParameters as evidence
    private double logSumProbability(String[] words, ClassParameters classParameters) {
        Double conditionalProbability = 0D;
        HashMap<String, Double> probabilityTable = classParameters.probabilityTable;
        // calculates conditional probability of a text, word by word, for a given class
        for (String word : words) {
            Double wordProbability = probabilityTable.get(word.toLowerCase());
            if (wordProbability != null) {
                conditionalProbability += Math.log(wordProbability);
            } else {
                // part of the laplace smoothing, use add 1 to all instead of 0 to avoid undefined logarithm
                conditionalProbability += Math.log((1 / (double) (classParameters.wordsCount + possibleWords)));
            }
        }
        return conditionalProbability;
    }

    /**
     * This class defines probability distribution of words in some user defined class/label.
     */
    private static class ClassParameters {
        private final String label; // label used as a result of classification
        private final HashMap<String, Double> probabilityTable; // probability of each word in this class
        private final double classProbability; // probability of the class itself observed from it's share in learning samples
        private final int wordsCount; // number of all words multiplied by their frequencies in this class

        private ClassParameters(String label, HashMap<String, Double> probabilityTable, double classProbability, int wordsCount) {
            this.label = label;
            this.probabilityTable = probabilityTable;
            this.classProbability = classProbability;
            this.wordsCount = wordsCount;
        }
    }
}
