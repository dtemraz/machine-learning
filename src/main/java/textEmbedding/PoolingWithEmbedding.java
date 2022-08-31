package textEmbedding;

import lombok.RequiredArgsConstructor;
import textEmbedding.embeddings.WordVectorizer;
import textEmbedding.pooling.Pooling;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This class combines pooling and embedding operations under single transform method.
 * Dimensions are defined inside {@link WordVectorizer}, this class therefore assumes that {@link Pooling} will return
 * vector of dimension equal embedding space dimension.
 */
@RequiredArgsConstructor
public class PoolingWithEmbedding {

    private final Pooling pooling;
    private final WordVectorizer wordVectorizer;

    /**
     * Returns configured pooling operation applied on embedded <em>tokens</em>.
     *
     * @param tokens for which to apply embedding pooling operations
     * @return configured pooling operation applied on embedded <em>tokens</em>
     */
    public double[] transform(String[] tokens) {
        double[][] embeddings = wordVectorizer.embed(tokens);
        return pooling.apply(embeddings);
    }

    public Map<Double, List<double[]>> transform(Map<Double, List<String[]>> data) {
        return data.entrySet().stream().parallel().collect(Collectors.toMap(Map.Entry::getKey,
                                                                            e -> e.getValue().stream()
                                                                                  .map(wordVectorizer::embed)
                                                                                  .map(pooling::apply)
                                                                                  .toList()));
    }

    /**
     * Returns number of dimensions in which tokens are embedded and result is pooled.
     * Method assumes that pooling operation reduces token vectors to a single vector inside same embedding space(has same
     * dimensionality).
     *
     * @return number of dimensions in which tokens are embedded and result is pooled
     */
    public int getDimensions() {
        return wordVectorizer.getDimensions();
    }

}
