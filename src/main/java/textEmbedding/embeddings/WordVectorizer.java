package textEmbedding.embeddings;

import java.io.Serializable;

/**
 * This interface defines word vectorisation. The interface lets user embed a single or multiple words with {@link #embed(String)}
 * and {@link #embed(String[])}.
 * User may also get information about dimensionality of embedding vector space via {@link #getDimensions()},
 */
public interface WordVectorizer {

    /**
     * Returns vector representation for <em>word</em> in configured embedding space.
     *
     * @param word for which to compute vector representation
     * @return vector representation for <em>word</em> in configured embedding space
     */
    double[] embed(String word);

    /**
     * Returns vector representation for all <em>words</em> in order they are defined, in configured embedding space.
     *
     * @param words for which to compute vector representation
     * @return vector representation for all <em>words</em> in order they are defined, in configured embedding space
     */
    double[][] embed(String[] words);

    /**
     * Returns dimensionality of configured embedding space.
     *
     * @return dimensionality of configured embedding space
     */
    int getDimensions();

}
