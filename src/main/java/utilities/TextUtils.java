package utilities;

import java.util.regex.Pattern;

/**
 * @author dtemraz
 */
public class TextUtils {

    private static Pattern punctuations = Pattern.compile("[.^,!?$]");

    public static boolean hasArabicCharacters(String text) {
        for (char c : text.toCharArray()) {
            if (Character.UnicodeBlock.of(c) == Character.UnicodeBlock.ARABIC) {
                return true;
            }
        }
        return false;
    }

    public static String removePunctuations(String text) {
       return punctuations.matcher(text).replaceAll(" ");
    }

}
