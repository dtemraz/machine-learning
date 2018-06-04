package utilities.text;

import java.util.regex.Pattern;

/**
 * @author dtemraz
 */
public class TextUtils {

    private static final Pattern PUNCTUATIONS = Pattern.compile("[.^,@!?$\\-]");
    private static final Pattern PUNCTUATIONS_AND_NEWLINE = Pattern.compile("[.^,@!?$\\-\n]|\r\n");

    /**
     * Returns true if the <em>text</em> contains character(s) in arabic unicode range, false otherwise.
     *
     * @param  text to check if contains arabic unicode characters
     * @return true if the text contains character(s) in arabic unicode range, false otherwise
     */
    public static boolean hasArabicCharacters(String text) {
        for (char c : text.toCharArray()) {
            if (Character.UnicodeBlock.of(c) == Character.UnicodeBlock.ARABIC) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns true if <em>word</em> contains digit and non-digit characters, false otherwise. This will considers words with
     * digits and special characters as alphanumeric even if there are no alphabet characters.
     *
     * @param word to check if it alphanumeric
     * @return true if <em>word</em> contains digit and non-digit characters, false otherwise
     */
    public static boolean isAlphaNumericWord(String word) {
        boolean hasNonNumeric = false;
        boolean hasNumbers = false;
        for (char ch : word.toCharArray()) {
            if (Character.isDigit(ch)) {
                hasNumbers = true;
            } else {
                hasNonNumeric = true;
            }
        }
        return hasNonNumeric && hasNumbers;
    }

    /**
     * Returns <em>true</em> if <em>word</em> contains special characters other than apostrophe('), <em>false</em>otherwise. The method will not
     * inspect last character, so if the <em>word</em> has a special character only in the <strong>last</strong>position the method will return false.
     *
     * <p>Apostrophe is excluded since there are many regular words such as don't which contain apostrophe.</p>
     *
     * @param word to check if it contains contains special characters other than apostrophe(')
     * @return true if <em>word</em> contains combination of special characters and letters or digits, false otherwise
     */
    public static boolean hasSpecialCharacters(String word) {
        char[] chars = word.toCharArray();
        // it is kinda okay if a word ends with special character if it is the only special character in the word
        for (int i = 0; i < chars.length - 1; i++) {
            char c = chars[i];
            // words with apostrophe are legit words most of the time
            if (!Character.isLetterOrDigit(c) && c != '\'') {
                return true;
            }
        }
        return false;
    }

    public static String removePunctuations(String text) {
        return deleteMatch(text, PUNCTUATIONS);
    }

    public static String removePunctuationsAndNewline(String text) {
        return deleteMatch(text, PUNCTUATIONS_AND_NEWLINE);
    }

    private static String deleteMatch(String text, Pattern pattern) {
        return pattern.matcher(text).replaceAll("");
    }

}