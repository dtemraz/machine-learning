package algorithms.bayes;

import algorithms.model.TextModel;
import lombok.ToString;
import structures.text.Vocabulary;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * This class implements <em>Multinomial Naive Bayes algorithm</em> with Laplace smoothing for text classification.
 * The algorithm uses Bayesian theorem: P(A|B) = (P(B|A) * P(A)) / P(B) where:
 * <ul>
 * <li>P(A|B) posterior: conditional probability of event A, given event B </li>
 * <li>P(B|A) likelihood: conditional probability of event B, given event A</li>
 * <li>P(A) prior: independent probability of event A</li>
 * <li>P(B) evidence: independent probability of event B</li>
 * </ul>
 * <p>
 * The algorithm is referred to as naive because it assumes there are no dependencies between features which simply doesn't hold
 * for most real world scenarios. For text analysis, it assumes that distribution of a given word doesn't affect distribution(existence)
 * of other words. Regardless, even with this naive assumption algorithm works reasonably well for many real word scenarios.
 *
 * <p>
 * The algorithm is trained via constructor with samples and their associated class labels provided with {@link Map}.
 * The user may get most probable class classId for a text via method {@link #classify(String)} once constructor finishes.
 * </p>
 * This algorithm is a solid choice when there are not many samples to learn from. While it may have greater asymptotic error
 * than it's counterpart linear regression model, it approaches the error faster:
 * <p>https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf</p>
 *
 * <p><strong>Corner cases:</strong></p>
 * <ul>
 * <li>probability of unseen words is handled with Laplace smoothing</li>
 * <li>numeric underflow is solved with logarithmic sum of probabilities instead of probabilities product</li>
 * </ul>
 *
 * @author dtemraz
 */
public class MultinomialNaiveBayes implements TextModel, Serializable {

    private static final long serialVersionUID = 1L;

    private static final String WHITESPACES = "\\s+"; // covers multiple whitespaces (tabs, new lines, spaces)

    private final List<ClassDistribution> classDistributions; // properties of each class defined with constructor samples

    /**
     * Constructs instance of this class trained to classify <em>samples</em> into classes defined with their labels.
     *
     * @param samples Map containing class id and associated texts to train Multinomial Naive Bayes algorithm
     */
    public MultinomialNaiveBayes(Map<Double, List<String[]>> samples) {
        this(samples, Vocabulary.NO_PRUNING);
    }

    /**
     * Constructs instance of this class trained to classify <em>samples</em> into classes defined with their labels, skipping all the words
     * in training phase which appear in less than <em>minCount</em> documents.
     *
     * @param samples  Map containing class id and associated texts to train Multinomial Naive Bayes algorithm
     * @param minCount minimal number of documents in which a word should appear to be used learning and classification
     */
    public MultinomialNaiveBayes(Map<Double, List<String[]>> samples, int minCount) {
        if (samples == null) {
            throw new IllegalArgumentException("samples must not be null");
        }
        classDistributions = train(samples, minCount);
    }

    /**
     * Returns the class with maximal posterior probability for this text. Text will be split into words with
     * {@link #WHITESPACES} expression.
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
     * Returns the class with maximal posterior probability for this text.
     *
     * @param words to classify into one of the labels supplied via constructor
     * @return most probable class classId for this text
     */
    @Override
    public double classify(String[] words) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException("words must not be null or empty");
        }
        return maxPosteriorClass(words);
    }

    /*
     LEARNING
     -------------------------------------------------------------------------------
    */

    // trains naive algorithms.bayes from the provided samples
    private List<ClassDistribution> train(Map<Double, List<String[]>> samples, int minCount) {
        // skip all the words which do not appear in minimal number of documents across all classes
        Set<String> tooRareWords = Vocabulary.findRareWords(samples.values().stream().flatMap(List::stream).collect(Collectors.toList()), minCount);

        int documentsCount = samples.values().stream().map(List::size).reduce(0, Integer::sum);

        Set<Double> classIds = samples.keySet();
        Map<Double, HashMap<String, Double>> frequencyTables = new HashMap<>();
        for (Double classId : classIds) {
            frequencyTables.put(classId, buildFrequencyTable(samples.get(classId), tooRareWords));
        }

        // count of unique words across all classes
        Set<String> uniqueWords = new HashSet<>();
        frequencyTables.values().forEach(t -> uniqueWords.addAll(t.keySet()));
        int allUniqueWords = uniqueWords.size();

        ArrayList<ClassDistribution> distributions = new ArrayList<>();
        for (Double classId : classIds) {
            List<String[]> classSamples = samples.get(classId);
            distributions.add(computeClassDistribution(classId, classSamples, frequencyTables.get(classId), documentsCount, allUniqueWords));
        }

        distributions.forEach(cd -> laplaceSmoothing(cd.probabilityTable, allUniqueWords));
        return distributions;
    }


    private ClassDistribution computeClassDistribution(Double classId, List<String[]> classSamples, HashMap<String, Double> ft, int documentsCount, int allUniqueWords) {
        double classProbability = Math.log((double) classSamples.size() / documentsCount);
        int wordsInClass = countWords(ft);
        double unseenWordProbability = Math.log((1 / (double) (wordsInClass + allUniqueWords)));
        return new ClassDistribution(classId, ft, classProbability, wordsInClass, unseenWordProbability);
    }

    // builds frequency table of words from the provided list of texts
    private HashMap<String, Double> buildFrequencyTable(List<String[]> texts, Set<String> tooRareWords) {
        HashMap<String, Double> frequencyTable = new HashMap<>();
        // ignore words which are to rare and have no information value
        texts.stream().flatMap(Arrays::stream).filter(word -> !tooRareWords.contains(word))
                .forEach(word -> frequencyTable.merge(word, 1D, (old, n) -> old + n));
        return frequencyTable;
    }

    // solves problem of missing words probability and numeric underflow with logarithmic function
    private void laplaceSmoothing(HashMap<String, Double> frequencyTable, int allUniqueWords) {
        // sum of words and their frequencies given the frequency table
        int totalWordsInClass = countWords(frequencyTable);
        // (X + a) / ( N + ad) , a = 1 for Laplace smoothing
        // add one to each frequency in the table and increase probability denominator by number of possible words
        for (Map.Entry<String, Double> entry : frequencyTable.entrySet()) {
            // transform frequencies into logarithmic scale, logarithm is monotonic function so the relative order is preserved
            entry.setValue(Math.log((entry.getValue() + 1) / (totalWordsInClass + allUniqueWords)));
        }
    }

    // returns total number of words in a frequency table
    private int countWords(HashMap<String, Double> frequencyTable) {
        Double wordsInClass = frequencyTable.entrySet().stream()
                .mapToDouble(Map.Entry::getValue)
                .reduce(0D, Double::sum);
        return wordsInClass.intValue();
    }

    /*
     INFERENCE
     -------------------------------------------------------------------------------
     */

    // returns class which is best explained by the available evidence (text)
    private double maxPosteriorClass(String[] text) {
        double max = Double.NEGATIVE_INFINITY;
        double label = -1;
        // this is just a standard find max algorithm
        for (ClassDistribution classDistribution : classDistributions) {
            // logarithmic scale, prior probability is not accounted for as it is the same for all classes
            double posterior = classDistribution.prior + likelihood(text, classDistribution);
            if (posterior > max) {
                max = posterior;
                label = classDistribution.classId;
            }
        }
        return label;
    }

    // returns sum of logarithmic probabilities for each word in a given class
    private double likelihood(String[] words, ClassDistribution classDistribution) {
        Double conditionalProbability = 0D;
        HashMap<String, Double> probabilityTable = classDistribution.probabilityTable;
        // calculates conditional probability of each word for a given class
        for (String word : words) {
            Double wordProbability = probabilityTable.get(word);
            // we are using + instead of * since Log(A*B) = Log(A) + Log(B)
            if (wordProbability != null) {
                conditionalProbability += wordProbability;
            } else {
                conditionalProbability += classDistribution.unseenWordsProbability;
            }
        }
        return conditionalProbability;
    }

    /**
     * This class defines probability distribution of words in some user defined class. The algorithm is trained individual
     * class distributions and in operative mode attempts to find the class that fits the text with highest probability.
     */
    @ToString
    private class ClassDistribution implements Serializable {
        private final double classId; // classId used as a result of classification
        private final HashMap<String, Double> probabilityTable; // probability of each word in this class
        private final double prior; // probability of the class itself being observed
        private final int wordsCount; // total number of (non-unique)words in a class
        private final double unseenWordsProbability; // use laplace smoothing for unseen words to avoid logarithm in zero

        private ClassDistribution(double classId, HashMap<String, Double> probabilityTable, double prior, int wordsCount, double unseenWordsProbability) {
            this.classId = classId;
            this.probabilityTable = probabilityTable;
            this.prior = prior;
            this.wordsCount = wordsCount;
            this.unseenWordsProbability = unseenWordsProbability;
        }
    }

}
