package structures.text;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;

/**
 * This class tests that vocabulary correctly computes {@link Term} objects with unique indexes and inverse document frequencies.
 *
 * @author dtemraz
 */
public class VocabularyTest {

    private Supplier<List<String[]>> documentsSupplier;
    private static final double delta = 0.00000001;

    @Before
    public void setUp() {
        documentsSupplier = new DocumentsSupplier()::get;
    }

    @After
    public void tearDown() {
        documentsSupplier = null;
    }

    /**
     * This tests checks that inverse document frequency is correctly calculated. Words appearing multiple times in a single
     * line should only contribute once per line to IDF score.
     */
    @Test
    public void idfShouldBeCorrectlyCalculated() {
        // given
        List<String[]> documents = documentsSupplier.get();
        Vocabulary vocabulary = new Vocabulary(documents);
        // when
        Term lifeTerm = vocabulary.get("life");
        Term gameTerm = vocabulary.get("game");
        // then
        assertEquals(1 + Math.log(documents.size() / 2.0), lifeTerm.idf, delta); // life in 2 out of 3 documents
        assertEquals(1 + Math.log(documents.size() / 1.0), gameTerm.idf, delta); // game in 1 out of 3 documents
    }

    /**
     * This tests checks that each term in vocabulary has unique id.
     */
    @Test
    public void termsShouldBeUnique() {
        // given
        Vocabulary vocabulary = new Vocabulary(documentsSupplier.get());
        // when
        Collection<Term> terms = vocabulary.getTerms();
        // then
        assertUniqueIds(terms);
    }

    /**
     * This tests checks if each unique word is present in vocabulary.
     */
    @Test
    public void allWordsShouldBeInVocabulary() {
        // given
        List<String[]> documents = documentsSupplier.get();
        Vocabulary vocabulary = new Vocabulary(documents);
        // then
        allWordsInVocabulary(vocabulary, documents);
    }


    private void assertUniqueIds(Collection<Term> terms) {
        assertEquals(terms.size(), terms.stream().map(Term::getId).distinct().count());
    }

    private void allWordsInVocabulary(Vocabulary vocabulary, List<String[]> documents) {
        long unknownWords = documents.stream().flatMap(Arrays::stream).filter(word -> vocabulary.get(word) == null).count();
        if (unknownWords > 0) {
            throw new AssertionError(String.format("there are: %d unknown words in vocabulary", unknownWords));
        }
    }

}
