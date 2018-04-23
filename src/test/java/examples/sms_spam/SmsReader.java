package examples.sms_spam;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * @author dtemraz
 */
public class SmsReader {

    private static final String SPAM = "spam";
    private static final String HAM = "ham";

    // TODO refactor, method should calculate stop words from data set based on their frequency
    private static final String STOP_WORDS = "\\ba\\b|\\bthe\\b|\\bu\\b|\\bin\\b|\\bis\\b|\\bme\\b|\\bmy\\b|\\bto\\b|\\byou\\b|\\band\\b|\\bI\\b|\\bi\\b";


    public Map<SmsCategory, LinkedList<String>> categorize(File file, boolean pruneStopWords) {
        Map<SmsCategory, LinkedList<String>> smsCategorises = new HashMap<>();
        smsCategorises.put(SmsCategory.SPAM, new LinkedList<>());
        smsCategorises.put(SmsCategory.HAM, new LinkedList<>());
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = "";
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(SPAM)) {
                    smsCategorises.get(SmsCategory.SPAM).add(line.split(SPAM)[1].trim());
                } else {
                    smsCategorises.get(SmsCategory.HAM).add(line.split(HAM)[1].trim());
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("error reading sms file ", e);
        }
        return pruneStopWords ? pruneStopWords(smsCategorises) : smsCategorises;
    }

    private Map<SmsCategory, LinkedList<String>>  pruneStopWords(Map<SmsCategory, LinkedList<String>> texts) {
        Map<SmsCategory, LinkedList<String>> pruned = new HashMap<>();
        LinkedList<String> spams = texts.get(SmsCategory.SPAM);
        LinkedList<String> prunedSpam = new LinkedList<>();
        for (String t : spams) {
            prunedSpam.add(t.replaceAll(STOP_WORDS, " "));
        }
        pruned.put(SmsCategory.SPAM, prunedSpam);

        LinkedList<String> prunedHam = new LinkedList<>();
        LinkedList<String> hams = texts.get(SmsCategory.HAM);
        for (String t : hams) {
            prunedHam.add(t.replaceAll(STOP_WORDS, " "));
        }
        pruned.put(SmsCategory.HAM, prunedHam);

        return pruned;
    }
}
