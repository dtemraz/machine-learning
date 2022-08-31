package textEmbedding.embeddings;

import lombok.RequiredArgsConstructor;
import structures.text.Vocabulary;

/**
 * This class implements word embeddings that should be applicable to any loaded embeddings.
 * User may initialise the class with method {@link #load(String)} with a pat to embeddings file.
 * <strong>Note on memory usage: </strong>
 * <p>
 * There is alternate method {@link #load(String, Vocabulary)} which can reduce memory usage by remove embeddings
 * for all words not seen in training. Unseen words with similar embeddings to seen words will be missing in such model.
 * Should user wish to reuse same embeddings for multiple models, {@link EmbeddingsLoader#cache(String)} method
 * may be invoked prior to share embedding matrix between them.
 * </p>
 */
@RequiredArgsConstructor
public class WordEmbeddings implements WordVectorizer {

    private final EmbeddingMatrix embeddingMatrix;

    /**
     * Returns {@link WordEmbeddings} vectorizers with embedding matrix loaded from file.
     * If multiple models use same embeddings, user may use {@link EmbeddingsLoader#cache(String)} to save
     * memory as embedding matrix will be shared between those.
     *
     * @param file from which to load embeddings
     * @return {@link WordEmbeddings} vectorizers with embedding matrix loaded from file
     */
    public static WordEmbeddings load(String file) {
        return load(file, null);
    }

    /**
     * Returns {@link WordEmbeddings} vectorizers with embedding matrix loaded from file. Additionally,
     * removes from the embedding matrix all words not present in the (training) vocabulary to save space.
     * <p>
     * If multiple models use same embeddings, user may use {@link EmbeddingsLoader#cache(String)} to save
     * memory as embedding matrix will be shared between those.
     * </p>
     *
     * @param file from which to load embeddings
     * @return {@link WordEmbeddings} vectorizers with embedding matrix loaded from file
     */
    public static WordEmbeddings load(String file, Vocabulary vocabulary) {
        return new WordEmbeddings(EmbeddingsLoader.loadEmbeddings(file, vocabulary));
    }

    @Override
    public double[] embed(String word) {
        return embeddingMatrix.embed(word);
    }

    @Override
    public double[][] embed(String[] tokens) {
        return embeddingMatrix.embed(tokens);
    }

    @Override
    public int getDimensions() {
        return embeddingMatrix.dimensions();
    }

}
