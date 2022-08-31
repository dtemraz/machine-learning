package examples.sms_spam;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * @author dtemraz
 */
public class SmsFeatureExtractor {

    private static final Pattern p = Pattern.compile("://");
    private static final String WHITESPACES = "\\s+"; // covers multiple whitespaces (tabs, new lines, spaces)

    public static String extractFeatures(String text) {
        Map<SmsMeta, Integer> metaInfo = getMetaInfo(text);
        StringBuilder metaBuilder = new StringBuilder(128);
        metaInfo.entrySet().forEach(e -> metaBuilder.append(formatMetaName(e.getKey(), e.getValue()) + " "));
        return metaBuilder.toString() + " " + removeSpecialCharacters(text);
    }

    private static Map<SmsMeta, Integer> getMetaInfo(String text) {
        LinkedHashMap<SmsMeta, Integer> meta = new LinkedHashMap<>();
        int[] symbols = characterPassThrough(text);
        meta.put(SmsMeta.DOLLAR, symbols[1]);
        meta.put(SmsMeta.NUMERIC, symbols[0]);
        meta.put(SmsMeta.HYPERLINK, symbols[2]);
        meta.put(SmsMeta.TOTAL_LENGTH, text.length());
        meta.put(lengthBucket(text), 1);
        return meta;
    }

    private static int[] characterPassThrough(String text) {
        int dollars = 0;
        int hyperLink = 0;
        char[] chars = text.toCharArray();
        for (char ch : chars) {
            if (ch == '$') {
                dollars++;
            }
        }
        Matcher hyperLinkMatcher = p.matcher(text);
        while (hyperLinkMatcher.find()) {
            hyperLink++;
        }
        // this will consider numeric strings instead of digits, try as well
        String[] words = text.trim().split(WHITESPACES);
        Long numericStrings = Stream.of(words).filter(SmsFeatureExtractor::isAlphaNumeric).count();
        return new int[]{numericStrings.intValue(), dollars, hyperLink};
    }

    private static String removeSpecialCharacters(String text) {
        return text.trim().toLowerCase().replaceAll("[.^,!?$]"," ");
    }

    private static boolean isAlphaNumeric(String text) {
        for (char ch : text.toCharArray()) {
            if (ch >= '0' && ch <= '9') {
                return true;
            }
        }
        return false;
    }

    private static SmsMeta lengthBucket(String text) {
        if (text.length() <= 40) {
            return SmsMeta.LENGTH_40;
        }
        if (text.length() <= 60) {
            return SmsMeta.LENGTH_60;
        }
        if (text.length() <= 80) {
            return SmsMeta.LENGTH_80;
        }
        if (text.length() <= 120) {
            return SmsMeta.LENGTH_120;
        }
        return SmsMeta.LENGTH_160;
    }

    private static String formatMetaName(SmsMeta smsMeta, Integer value) {
        return "_x_" + smsMeta.toString().toLowerCase() + "_" + value + "_x_";
    }

}
