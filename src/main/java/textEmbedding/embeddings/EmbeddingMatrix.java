package textEmbedding.embeddings;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;

/**
 * This class defines embedding matrix for arbitrary word embeddings. The matrix has embedding vector for each word,
 * dimensionality of vector space(size of embedding) and representation for unknown token.
 */
public class EmbeddingMatrix implements Serializable {

    public static final String UNKNOWN_TOKEN = "unk";

    private final HashMap<String, double[]> wordEmbeddings;
    private final double[] unknownEmbedding;
    private final int dimensions;

    /**
     * Creates instance of {@link EmbeddingMatrix} from supplied <em>wordEmbeddings</em> which must contain key: "unk"
     * to embed unknown tokens.
     *
     * @param wordEmbeddings a map of words and their embeddings
     * @throws IllegalArgumentException if <em>wordEmbeddings</em> do not contain key: "unk"; an embedding for unknown words
     */
    public EmbeddingMatrix(HashMap<String, double[]> wordEmbeddings) {
        this.wordEmbeddings = wordEmbeddings;
        // unknownEmbedding = wordEmbeddings.get(UNKNOWN_TOKEN);
        // dimensions = unknownEmbedding.length;
        dimensions = 100;
        unknownEmbedding = new double[100];

        if (!wordEmbeddings.containsKey(UNKNOWN_TOKEN)) {
            // throw new IllegalArgumentException("embedding matrix must have embedding for unknown tokens with name: " + UNKNOWN_TOKEN);
        }
    }

    /**
     * Returns embedding vector for <em>token</em>.
     *
     * @param token for which to return embedding vector
     * @return embedding vector for <em>token</em>
     */
    public double[] embed(String token) {
        return wordEmbeddings.getOrDefault(token, unknownEmbedding);
    }

    /**
     * Returns embedding matrix for <em>tokens</em>.
     *
     * @param tokens for which to return embedding matrix
     * @return embedding matrix for <em>tokens</em>
     */
    public double[][] embed(String[] tokens) {
        double[][] embeddings = new double[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
            embeddings[i] = embed(tokens[i]);
        }
        return embeddings;
    }

    /**
     * Returns set of all tokens in embedding matrix.
     *
     * @return set of all tokens in embedding matrix
     */
    public Set<String> tokens() {
        return wordEmbeddings.keySet();
    }

    /**
     * Returns number of tokens in embedding matrix.
     *
     * @return number of tokens in embedding matrix
     */
    public int size() {
        return wordEmbeddings.size();
    }

    /**
     * Returns dimensionality of vector space(size of embeddings) for loaded word embeddings.
     *
     * @return dimensionality of vector space(size of embeddings) for loaded word embeddings
     */
    public int dimensions() {
        return dimensions;
    }

}
