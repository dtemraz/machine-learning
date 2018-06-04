package structures.text;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * @author dtemraz
 */
public class DocumentsSupplier implements Supplier<List<String[]>> {

    private static List<String[]> documents = getDocuments();

    @Override
    public List<String[]> get() {
        return documents;
    }

    /*
     * credit for verification samples: https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
     */

    private static ArrayList<String[]> getDocuments() {
        ArrayList<String[]> documents = new ArrayList<>();
        final String wordsDelimiter = " ";
        documents.add("The game of life is a game of everlasting learning".split(wordsDelimiter));
        documents.add("The unexamined life is not worth living".split(wordsDelimiter));
        documents.add("Never stop learning".split(wordsDelimiter));
        return documents;
    }
}
