package structures.text;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.*;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        // when
        Vocabulary vocabulary = new Vocabulary(documents);
        // then
        allWordsInVocabulary(vocabulary, documents);
    }

    /**
     * Checks that all words which appear in only one document are removed
     */
    @Test
    public void rareWordsShouldBePruned() {
        // given
        List<String[]> documents = documentsSupplier.get();
        // when
        Vocabulary vocabulary = new Vocabulary(documents, 2);
        // then
        assertPrunedVocabulary(vocabulary, documents.size());
    }

    /**
     * Checks that count of documents each word appears in is correct
     */
    @Test
    public void wordPresenceInDocumentsShouldBeCounted() {
        // given
        List<String[]> documents = documentsSupplier.get();
        // when
        HashMap<String, Double> wordPresences = Vocabulary.countWords(documents);
        // then
        assertWordPresences(wordPresences);
    }

    /**
     * Checks that all words which appear in only one document are found
     */
    @Test
    public void wordsThatAppearInOneDocumentShouldBeFound() {
        // given
        List<String[]> documents = documentsSupplier.get();
        // when
        int minDocuments = 2;
        Set<String> rareWords = Vocabulary.findRareWords(documents, minDocuments);
        // then all single document words should be found
        assertTrue(singleDocumentWords().stream().filter(rareWords::contains).count() == singleDocumentWords().size());
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

    private void assertPrunedVocabulary(Vocabulary vocabulary, int documents) {
        // words: {the life is learning} are present in 2 documents
        int remainingWords = 4;
        assertEquals(remainingWords, vocabulary.size());
        // each term should have unique id and they should be assigned from 0 to size
        int[] termIds = vocabulary.getTerms().stream().mapToInt(Term::getId).sorted().toArray();
        for (int id = 0; id < remainingWords; id++) {
            assertEquals(id, termIds[id]);
        }
        // each remaining term appears in exactly two documents
        double documentsContainingWord = 2;
        for (Term t : vocabulary.getTerms()) {
            assertEquals(1 + Math.log(documents / documentsContainingWord), t.idf, delta); // life in 2 out of 3 documents
        }
    }

    private void assertWordPresences(HashMap<String, Double> wordPresences) {
        List<String> singleDocumentWords = singleDocumentWords();
        List<String> twoDocumentWords = twoDocumentWords();
        // all words should be accounted for
        assertEquals(singleDocumentWords.size() + twoDocumentWords.size(), wordPresences.size());
        singleDocumentWords.forEach(w -> assertEquals(1D, wordPresences.get(w), 0));
        twoDocumentWords.forEach(w -> assertEquals(2D, wordPresences.get(w), 0));
    }

    private List<String> singleDocumentWords() {
        return Arrays.asList("game", "of", "a", "everlasting", "unexamined", "not", "worth", "living", "stop", "Never");
    }

    private List<String> twoDocumentWords() {
        return Arrays.asList("The", "life", "is", "learning");
    }

}
