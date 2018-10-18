package utilities.text;

/**
 * @author dtemraz
 */
public class LanguageUtils {

    private static final int ASCII_RADIX = 128;
    private static final UnicodeBlock MYANMAR_BASE = new UnicodeBlock(0x1000, 0x109F);
    private static final UnicodeBlock MYANMAR_EXTENDED_A = new UnicodeBlock(0xAA60, 0xAA7F);
    private static final UnicodeBlock MYANMAR_EXTENDED_B = new UnicodeBlock(0xA9E0, 0xA9FF);

    /**
     * Returns true if <strong>all</strong> characters in text are in ASCII alphabet, false otherwise.
     *
     * @param text to check if in ASCII
     * @return true if all characters in text are in ASCII alphabet, false otherwise
     */
    public static boolean inAsciiAlphabet(String text) {
        for (char c : text.toCharArray()) {
            if (c >= ASCII_RADIX) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns <em>true</em> if there is at least one character in text from Myanmar Unicode block, <em>false</em> otherwise.
     *
     * <p>The method will stop on first Myanmar character encountered, if there are any.</p>
     *
     * @param text to check for Myanmar characters
     * @return <em>true</em> if there is at least one character in text from Myanmar Unicode block, <em>false</em> otherwise
     */
    public static boolean hasMyanmarCharacter(String text) {
        for (char c : text.toCharArray()) {
            if (MYANMAR_BASE.contains(c) || MYANMAR_EXTENDED_A.contains(c) || MYANMAR_EXTENDED_B.contains(c)) {
                return true;
            }
        }
        return false;
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
