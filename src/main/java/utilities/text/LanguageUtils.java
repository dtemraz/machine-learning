package utilities.text;

import java.text.Normalizer;
import java.util.regex.Pattern;

/**
 * @author dtemraz
 */
public class LanguageUtils {

    private static final int ASCII_RADIX = 128;
    private static final int EXTENDED_ASCII_RADIX = 256;

    private static final Pattern ASCII = Pattern.compile("[^\\p{ASCII}]"); // replace everything that is not ascii

    /**
     * Strips accents from the text using canonical decomposition(NFD) and replaces them with their ASCII counterparts.
     * All non ascii characters will be removed from the <em>text</em>.
     *
     * @param text from which to strip accents
     * @return text stripped of accents and with only ascii characters
     */
    public static String stripAccents(String text) {
        if (text == null) {
            return null;
        }
        return ASCII.matcher(Normalizer.normalize(text, Normalizer.Form.NFD)).replaceAll("");
    }

    /**
     * Returns <em>true</em> if <strong>all</strong> characters in text are in ASCII alphabet, <em>false</em> otherwise.
     *
     * @param text to check if in ASCII
     * @return <em>true</em> if all characters in text are in ASCII alphabet, <em>false</em> otherwise
     */
    public static boolean inAsciiAlphabet(String text) {
        return inAsciiAlphabet(text, ASCII_RADIX);
    }

    /**
     * Returns <em>true</em> if <strong>all</strong> characters in text are in EXTENDED ASCII alphabet, <em>false</em> otherwise.
     *
     * @param text to check if in ASCII
     * @return <em>true</em> if all characters in text are in ASCII alphabet, <em>false</em> otherwise
     */
    public static boolean inExtendedAsciiAlphabet(String text) {
        return inAsciiAlphabet(text, EXTENDED_ASCII_RADIX);
    }

    // checks if entire text is in ascii, or extended ascii alphabet, depending on the radix
    private static boolean inAsciiAlphabet(String text, int radix) {
        for (char c : text.toCharArray()) {
            if (c >= radix) {
                return false;
            }
        }
        return true;
    }


    /**
     * Returns <em>true</em> if there is at least one character in text from ASCII alphabet, <em>false</em> otherwise.
     *
     * <p>The method will stop on first ASCII character encountered, if there are any.</p>
     *
     * @param text to check for ASCII characters
     * @return <em>true</em> if there is at least one character in text from ASCII alphabet, <em>false</em> otherwise
     */
    public static boolean hasAsciiCharacter(String text) {
        for (char c : text.toCharArray()) {
            if (c != ' ' && c < ASCII_RADIX) {
                return true;
            }
        }
        return false;
    }

}