package utilities;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * This class offers utility methods for {@link Map} interface implementations. There is a method {@link #removeNumbers(Map)} which removes all digit only words from a map,
 * and method {@link #removeNumbersAndSingleLetterWords(Map)} which removes both numbers <strong>and</strong> single letter words.
 *
 * @author dtemraz
 */
public class MapUtil {

    private static final Pattern ONLY_DIGITS = Pattern.compile("\\d+");

    /**
     * Removes all digit only words in <em>dataSet</em>. If the resulting array of features has no other features, then it is removed
     * from the list.
     *
     * @param dataSet from which to remove digit only words
     */
    public static void removeNumbers(Map<Double, List<String[]>> dataSet) {
        dataSet.keySet().forEach(k -> dataSet.replace(k, removeNumbers(dataSet.get(k))));
    }

    /**
     * Removes all digit only words in <em>dataSet</em> and words which consists of a single letter only. If the resulting array of features has no other features, then it is removed
     * from the list.
     *
     * @param dataSet from which to remove digit only words and single letter words
     */
    public static void removeNumbersAndSingleLetterWords(Map<Double, List<String[]>> dataSet) {
        dataSet.keySet().forEach(k -> dataSet.replace(k, removeNumbersAndSingleLetterWords(dataSet.get(k))));
    }

    private static List<String[]> removeNumbers(List<String[]> features) {
        return features.stream().map(fts -> removeNumbers(fts, s -> !ONLY_DIGITS.matcher(s).matches())).filter(Objects::nonNull).collect(Collectors.toList());
    }

    private static List<String[]> removeNumbersAndSingleLetterWords(List<String[]> features) {
        return features.stream().map(fts -> removeNumbers(fts, s -> !ONLY_DIGITS.matcher(s).matches() && s.length() > 1)).filter(Objects::nonNull).collect(Collectors.toList());
    }

    // returns array without digit only features, or null if these were the only features
    private static String[] removeNumbers(String[] features, Predicate<String> toKeep) {
        List<String> withoutNumbers = new ArrayList<>();
        for (String feature : features) {
            if (toKeep.test(feature)) {
                withoutNumbers.add(feature);
            }
        }
        return withoutNumbers.size() > 0 ? withoutNumbers.toArray(new String[withoutNumbers.size()]) : null;
    }
}
