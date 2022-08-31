package textEmbedding.embeddings;

import lombok.RequiredArgsConstructor;
import structures.text.TF_IDF_Vectorizer;
import structures.text.Term;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.HashMap;

/**
 * Word embeddings when passed to downstream task are all equally important. This class lets user weight their importance
 * with TF IDF statistics computed from training set.
 * Method {@link #embed(String)} which takes a single word input does not apply any weighting.
 */
@RequiredArgsConstructor
public class WeightedWordEmbeddings implements WordVectorizer {

    private final Vocabulary vocabulary;
    private final WordVectorizer wordVectorizer;

    @Override
    public double[] embed(String word) {
        return wordVectorizer.embed(word);
    }

    // weight each embedding by the tf_idf to assign different importance to each word
    @Override
    public double[][] embed(String[] words) {
        // strictly speaking user may send single input words here
        double[][] embeddings = wordVectorizer.embed(words);
        // this will only return frequencies for words in vocabulary
        HashMap<Term, Double> termFrequencies = TF_IDF_Vectorizer.termFrequencies(words, vocabulary);
        // we can apply average tf idf for words outside training vocabulary
        double averageTfIdf = TF_IDF_Vectorizer.averageTfIdf(termFrequencies);
        for (int i = 0; i < words.length; i++) {
            Term term = vocabulary.get(words[i]);
            // we might have embedding but not seen the word in downstream task vocabulary, hence do not know idf
            if (!termFrequencies.containsKey(term)) {
                embeddings[i] = Vector.multiply(embeddings[i], averageTfIdf);
            } else {
                Double tf = termFrequencies.get(term);
                embeddings[i] = Vector.multiply(embeddings[i], tf * term.getIdf());
            }

        }
        return embeddings;
    }

    @Override
    public int getDimensions() {
        return wordVectorizer.getDimensions();
    }

}
