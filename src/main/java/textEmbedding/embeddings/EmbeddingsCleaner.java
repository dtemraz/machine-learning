package textEmbedding.embeddings;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class EmbeddingsCleaner {

    private static final Pattern WHITESPACE_SEPARATOR = Pattern.compile(" ");

    private final Predicate<char[]> keepWord;
    private final boolean keepHeader;
    private int cnt = 0;

    public EmbeddingsCleaner(boolean keepHeader) {
        this(false, true, true, 2);
    }

    public EmbeddingsCleaner(boolean keepHeader, boolean hasUpper, boolean hasAscii, int minSize) {
        // built filter will keep word if it passes all reject predicates, hence negate for reject
        this.keepWord = prepareFilters(hasUpper, hasAscii, minSize).stream()
                                                                   .reduce(f -> false, Predicate::or)
                                                                   .negate();
        this.keepHeader = keepHeader;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            throw new IllegalArgumentException("source file and target file must be defined as first two parameters");
        }
        EmbeddingsCleaner cleaner = new EmbeddingsCleaner(false, true, true, 2);
        String source = args[0];
        String target = args[1];
        cleaner.filterEmbeddings(source, target);
    }

    public void filterEmbeddings(String source, String target) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(target))) {
            if (keepHeader) {
                writeHeader(pw, source);
            }
            try (Stream<String> stream = Files.lines(Paths.get(source))) {
                stream.forEach(s -> copy(s, pw));
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("size: " + cnt);
        }
    }

    private void writeHeader(PrintWriter pw, String source) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(source))) {
            pw.println(reader.readLine());
        }
    }

    // copy a row to the target file if word passes configured filters
    private void copy(String row, PrintWriter pw) {
        String[] wordEmbedding = WHITESPACE_SEPARATOR.split(row);
        String w = wordEmbedding[0];
        char[] chars = w.toCharArray();
        if (keepWord.test(chars)) {
            cnt++;
            pw.println(row);
        }
    }

    // there are a lot of dates, random numbers and special characters

    private boolean hasLetters(char[] chars) {
        for (char c : chars) {
            if (Character.isLetter(c)) {
                return true;
            }
        }
        return false;
    }

    // bunch of numbers and dates
    private boolean hasDigits(char[] chars) {
        for (char c : chars) {
            if (Character.isDigit(c)) {
                return true;
            }
        }
        return false;
    }

    // for english embeddings, these are typically non english words :)
    private boolean hasNonAscii(char[] chars) {
        for (char c : chars) {
            if (c > 127) {
                return true;
            }
        }
        return false;
    }

    // most of the words with upper case are company names, street addresses, names..
    private boolean hasUpper(char[] chars) {
        for (char c : chars) {
            if (Character.isUpperCase(c)) {
                return true;
            }
        }
        return false;
    }

    // create reject filters to shrink embedding matrix
    private List<Predicate<char[]>> prepareFilters(boolean hasUpper, boolean hasNonAscii, int minSize) {
        List<Predicate<char[]>> cleaners = new ArrayList<>();
        cleaners.add(s -> s.length < minSize);
        cleaners.add(Predicate.not(this::hasLetters));
        cleaners.add(this::hasDigits);
        if (hasUpper) {
            cleaners.add(this::hasUpper);
        }
        if (hasNonAscii) {
            cleaners.add(this::hasNonAscii);
        }
        return cleaners;
    }

}
