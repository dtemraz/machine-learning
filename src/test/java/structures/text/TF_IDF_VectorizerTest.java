package structures.text;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;

/**
 * This class verifies that {@link TF_IDF_Vectorizer} correctly calculates TF and TF-IDF values of terms in a message.
 *
 * @author dtemraz
 */
public class TF_IDF_VectorizerTest {

    private Supplier<List<String[]>> documentsSupplier;
    private Vocabulary vocabulary;
    private static final double delta = 0.00000001;

    @Before
    public void setUp() {
        documentsSupplier = new DocumentsSupplier()::get;
        vocabulary = new Vocabulary(documentsSupplier.get());
    }

    @After
    public void tearDown() {
        documentsSupplier = null;
    }

    /**
     * This tests checks that frequency of each term(word) in a message is correctly calculated.
     */
    @Test
    public void termFrequenciesShouldBeCorrectlyCalculated() {
        // given
        String[] message = documentsSupplier.get().get(0); // The game of life is a game of everlasting learning"
        // when
        HashMap<Term, Double> termFrequencies = TF_IDF_Vectorizer.termFrequencies(message, vocabulary);
        // then
        assertTermFrequencies(termFrequencies, message);
    }

    /**
     * This tests checks that frequency of each term(word) in a message is correctly calculated.
     */
    @Test
    public void tfIdfShouldBeCorrectlyCalculated() {
        // given
        String[] message = documentsSupplier.get().get(0); // The game of life is a game of everlasting learning"
        // when
        TF_IDF_Term[] tf_idf_terms = TF_IDF_Vectorizer.tfIdf(message, vocabulary);
        // then
        assertTfIdf(tf_idf_terms);
    }


    private void assertTermFrequencies(HashMap<Term, Double> termFrequencies, String[] message) {
        HashMap<String, Double> expectedFrequencies = new HashMap<>();
        expectedFrequencies.put("The", 1d / message.length);
        expectedFrequencies.put("game", 2d / message.length);
        expectedFrequencies.put("of", 2d / message.length);
        expectedFrequencies.put("life", 1d / message.length);
        expectedFrequencies.put("is", 1d / message.length);
        expectedFrequencies.put("a", 1d / message.length);
        expectedFrequencies.put("everlasting", 1d / message.length);
        expectedFrequencies.put("learning", 1d / message.length);
        for (String word : message) {
            Double frequency = termFrequencies.get(vocabulary.get(word));
            assertEquals(expectedFrequencies.get(word), frequency, delta); // life in 2 out of 3 documents
        }
    }

    private void assertTfIdf(TF_IDF_Term[] tf_idf_terms) {
        HashMap<String, Double> expectedTfIdf = new HashMap<>();
        String life = "life";
        String game = "game";
        expectedTfIdf.put(life, (1d / 10) * (1 + Math.log(3 / 2.0))); // life in 2 out of 3 documents
        expectedTfIdf.put(game, (2d / 10) * (1 + Math.log(3 / 1.0))); // game in 1 out of 3 documents

        TF_IDF_Term lifeTerm = Arrays.stream(tf_idf_terms).filter(t -> t.getId() == vocabulary.get(life).getId()).findFirst().get();
        assertEquals(expectedTfIdf.get(life), lifeTerm.getTfIdf(), delta);

        TF_IDF_Term gameTerm = Arrays.stream(tf_idf_terms).filter(t -> t.getId() == vocabulary.get(game).getId()).findFirst().get();
        assertEquals(expectedTfIdf.get(game), gameTerm.getTfIdf(), delta);

    }

}
