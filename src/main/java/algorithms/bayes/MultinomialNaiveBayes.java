package algorithms.bayes;

import algorithms.ensemble.model.TextModel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

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
 * The algorithm is trained via constructor with samples and their associated class labels provided with {@link Map}.
 * The user may get most probable class classId for a text via method {@link #classify(String)} once constructor finishes.
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
public class MultinomialNaiveBayes implements TextModel, Serializable {

    private static final long serialVersionUID = 1L;

    private static final String WHITESPACES = "\\s+"; // covers multiple whitespaces (tabs, new lines, spaces)

    private final List<ClassDistribution> classDistributions; // properties of each class defined with constructor samples
    private final int possibleWords; // count of unique set of possible words from all classes

    /**
     * Constructs instance of this class trained to classify <em>samples</em> into classes defined with their labels.
     *
     * @param samples Map containing class id and associated texts to train Multinomial Naive Bayes algorithm
     */
    public MultinomialNaiveBayes(Map<Double, List<String[]>> samples) {
        if (samples == null) {
            throw new IllegalArgumentException("samples must not be null");
        }
        classDistributions = train(samples);
        // we need this for laplace smoothing since we are effectively raising count of all possible words(vocabulary) by 1
        possibleWords = uniqueWordsCount();
        // handle unseen words since they would have probability 0 which would cause problems for logarithmic function
        classDistributions.forEach(cd -> laplaceSmoothing(cd.probabilityTable));
    }

    /**
     * Returns the most likely class classId for this text according to Bayesian theorem. Text will be split into words
     * with {@link #WHITESPACES} expression.
     *
     * @param text to classify into one of the labels supplied via constructor
     * @return most probable class classId for this text
     */
    public double classify(String text) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("text must not be null or empty");
        }
        return classify(text.split(WHITESPACES));
    }

    /**
     * Returns the most likely class classId for these words according to Bayesian theorem.
     *
     * @param words to classify into one of the labels supplied via constructor
     * @return most probable class classId for this text
     */
    public double classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        return maxLikelihoodClass(words);
    }

    /*
    Learning
    -------------------------------------------------------------------------------
   */

    // trains naive algorithms.bayes from the provided samples
    private List<ClassDistribution> train(Map<Double, List<String[]>> samples) {
        // sums sizes of texts in each sample
        int totalSamples = samples.values().stream().map(List::size).reduce(Integer::sum).get();
        // properties of each modeled class
        ArrayList<ClassDistribution> classParameters = new ArrayList<>();
        // learn class properties(distribution) of each class from samples
        samples.forEach((key, value) -> classParameters.add(getClassDistribution(value, key, totalSamples)));
        return classParameters;
    }

    // calculates words distribution parameters for the given textSamples of a class
    private ClassDistribution getClassDistribution(List<String[]> texts, double classId, int totalSamples) {
        HashMap<String, Double> frequencyTable = buildFrequencyTable(texts);
        double classProbability = Math.log((double) texts.size() / totalSamples);
        return new ClassDistribution(classId, frequencyTable, classProbability, countWords(frequencyTable));
    }

    // builds frequency table of words from the provided list of texts
    private HashMap<String, Double> buildFrequencyTable(List<String[]> texts) {
        HashMap<String, Double> frequencyTable = new HashMap<>(); // key = word, value = count
        texts.stream().flatMap(Arrays::stream).forEach(word -> frequencyTable.merge(word, 1D, (old, n) -> old + n));
        return frequencyTable;
    }

    // solves problem of missing words probability and numeric underflow with logarithmic function
    private void laplaceSmoothing(HashMap<String, Double> frequencyTable) {
        // sum of words and their frequencies given the frequency table
        int totalWordsInClass = countWords(frequencyTable);
        // add one to each frequency in the table and increase probability denominator by number of possible words
        frequencyTable.forEach((key, value) -> frequencyTable.compute(key, (term, frequency) -> Math.log((frequency + 1) / (totalWordsInClass + possibleWords))));
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
        classDistributions.forEach(cd -> uniqueWords.addAll(cd.probabilityTable.keySet()));
        return uniqueWords.size();
    }

    /*
     Classification
     -------------------------------------------------------------------------------
     */

    /**
     * Returns class with the highest probability of containing words in this <em>text</em>.
     * <p>
     * There is a hypothesis per each class where we assume that the words belong to a class. Hypothesis(class) with maximal
     * probability to match the text is chosen as the output class.
     * </p>
     *
     * @param text to classify
     * @return class id which most likely corresponds to this <em>text</em>
     */
    private double maxLikelihoodClass(String[] text) {
        double max = Double.NEGATIVE_INFINITY;
        double label = -1;
        // this is just a standard find max algorithm
        for (ClassDistribution classDistribution : classDistributions) {
            double probability = classDistribution.classProbability + logSumProbability(text, classDistribution);
            if (probability > max) {
                max = probability;
                label = classDistribution.classId;
            }
        }
        return label;
    }

    // returns sum of logarithmic probabilities for each word, given class defined with classDistribution as evidence
    private double logSumProbability(String[] words, ClassDistribution classDistribution) {
        Double conditionalProbability = 0D;
        HashMap<String, Double> probabilityTable = classDistribution.probabilityTable;
        // calculates conditional probability of a text, word by word, for a given class
        for (String word : words) {
            Double wordProbability = probabilityTable.get(word);
            // we are using + instead of * since Log(A*B) = Log(A) + Log(B)
            if (wordProbability != null) {
                conditionalProbability += wordProbability;
            } else {
                // laplace smoothing for unseen words - add 1 to avoid undefined logarithm in 0
                conditionalProbability += Math.log((1 / (double) (classDistribution.wordsCount + possibleWords)));
            }
        }
        return conditionalProbability;
    }

    /**
     * This class defines probability distribution of words in some user defined class. The algorithm is trained individual
     * class distributions and in operative mode attempts to find the class that fits the text with highest probability.
     */
    private class ClassDistribution implements Serializable {
        private final double classId; // classId used as a result of classification
        private final HashMap<String, Double> probabilityTable; // probability of each word in this class
        private final double classProbability; // probability of the class itself observed from it's share in learning samples
        private final int wordsCount; // number of all words multiplied by their frequencies in this class

        private ClassDistribution(double classId, HashMap<String, Double> probabilityTable, double classProbability, int wordsCount) {
            this.classId = classId;
            this.probabilityTable = probabilityTable;
            this.classProbability = classProbability;
            this.wordsCount = wordsCount;
        }
    }

}
