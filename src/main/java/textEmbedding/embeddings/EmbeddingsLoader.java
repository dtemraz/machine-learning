package textEmbedding.embeddings;

import structures.text.Vocabulary;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * This class lets user load and optionally cache word embedding matrix from the provided file.
 * At the moment only last loaded embedding matrix is cached, although it could be changed so there is a cache per embedding file.
 * Should user wish to use same embeddings for multiple downstream tasks, it is recommend to cache the embedding via call
 * to {@link #cache(String)}. This should result with significant memory savings.
 */
public class EmbeddingsLoader {
    private static final Pattern WHITESPACE_SEPARATOR = Pattern.compile(" ");
    private static EmbeddingMatrix cachedMatrix;
    private static String cachedEmbeddingsFile;

    public static void main(String[] args) {
        loadEmbeddings("/Users/dtemraz/IdeaProjects/machine_learning/src/test/resources/embeddings#100d.txt");
        System.out.println();
    }

    /**
     * This method lets user cache embedding matrix loaded from <em>file</em>. Once the method finishes, subsequent calls
     * to {@link #loadEmbeddings(String)} will return cached matrix.
     *
     * @param embeddingsFile from which to load embedding matrix
     */
    public static synchronized EmbeddingMatrix cache(String embeddingsFile) {
        cachedEmbeddingsFile = embeddingsFile;
        cachedMatrix = loadEmbeddings(embeddingsFile);
        return cachedMatrix;
    }

    /**
     * Returns embedding matrix loaded from <em>embeddingsFile</em>. If there is cached matrix for the <em>embeddingsFile</em>
     * generated with {@link #cache(String)}, object will be reused.
     *
     * @param embeddingsFile from which to load embedding matrix
     * @return embedding matrix
     */
    public static synchronized EmbeddingMatrix loadEmbeddings(String embeddingsFile) {
        // cached matrix should not be pruned with vocabulary
        if (embeddingsFile.equals(cachedEmbeddingsFile)) {
            return cachedMatrix;
        }
        return loadEmbeddings(embeddingsFile, null);
    }

    /**
     * Returns embedding matrix loaded from <em>embeddingsFile</em>. Matrix will only contain words present in <em>vocabulary</em>
     * + out of vocabulary token. This will likely reduce memory footprint but may have some impact on model performance.
     *
     * @param embeddingsFile from which to load embedding matrix
     * @param vocabulary containing words from training
     * @return embedding matrix
     */
    public static EmbeddingMatrix loadEmbeddings(String embeddingsFile, Vocabulary vocabulary) {
        HashMap<String, double[]> embeddingMatrix = new HashMap<>((int) Math.ceil(100_000 / 0.75));
        int dimensions = dimensions(embeddingsFile);
        try (Stream<String> stream = Files.lines(Paths.get(embeddingsFile))) {
            stream.forEach(s -> prepareEmbedding(embeddingMatrix, s, dimensions));
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (vocabulary != null) {
            deleteOOVTokens(embeddingMatrix, vocabulary);
        }
        return new EmbeddingMatrix(embeddingMatrix);
    }

    // delete all words that were not present in the training set
    private static void deleteOOVTokens(HashMap<String, double[]> embeddingMatrix, Vocabulary vocabulary) {
        for (String s : embeddingMatrix.keySet()) {
            if (!vocabulary.contains(s)) {
                embeddingMatrix.remove(s);
            }
        }
    }

    // add current word, and it's embedding to the embedding matrix
    private static void prepareEmbedding(HashMap<String, double[]> embeddingMatrix, String row, int dimensions) {
        String[] wordEmbedding = WHITESPACE_SEPARATOR.split(row, 2);
        String w = wordEmbedding[0];
        String[] e = WHITESPACE_SEPARATOR.split(wordEmbedding[1]);
        double[] embedding = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            embedding[i] = Double.parseDouble(e[i]);
        }
        embeddingMatrix.put(w, embedding);
    }

    private static int dimensions(String filename) {
        int d = filename.lastIndexOf('d');
        int hash = filename.lastIndexOf('#');
        return Integer.parseInt(filename.substring(hash + 1, d));
    }

}
