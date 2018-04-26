package examples.sms_spam;

import bayes.MultinomialNaiveBayes;
import bayes.TextSamples;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Sample results
 * overall accuracy : 0.9843566726511068
 * spams caught: 0.9278571428571426
 * hams blocked: 0.006897028334485141
 *
 * TODO ugly refactor
 *
 * @author dtemraz
 */
public class TestSmsSpam {

    static int runs = 50;

    static List<Double> overall = new ArrayList<>();
    static List<Double> spamsCaught = new ArrayList<>();
    static List<Double> blockedHams = new ArrayList<>();

    public static void main(String[] args) {
        SmsReader reader = new SmsReader();
        Map<SmsCategory, LinkedList<String>> smsCategories = reader.categorize(new File("src/test/resources/sms_spam.txt"), true);
        LinkedList<String> spamFeatures = smsCategories.get(SmsCategory.SPAM).stream().map(SmsFeatureExtractor::extractFeatures).collect(Collectors.toCollection(LinkedList::new));
        LinkedList<String> hamFeatures = smsCategories.get(SmsCategory.HAM).stream().map(SmsFeatureExtractor::extractFeatures).collect(Collectors.toCollection(LinkedList::new));

        for (int i = 0; i < runs; i++) {

            DataSet dataSet = stratification(new LinkedList<>(spamFeatures), new LinkedList<>(hamFeatures));
            TextSamples spam = new TextSamples("SPAM", dataSet.trainingSpam);
            TextSamples ham = new TextSamples("HAM", dataSet.trainingHam);
            MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(Arrays.asList(spam, ham));

            int spamPositive = 0;
            int spamNegative = 0;
            for (String message : dataSet.validationSpam) {
                if (mnb.classify(message).equals("SPAM")) {
                    spamPositive++;
                } else {
                    spamNegative++;
                }
            }

            int hamPositive = 0;
            int hamNegative = 0;
            for (String message : dataSet.validationHam) {
                if (mnb.classify(message).equals("HAM")) {
                    hamPositive++;
                } else {
                    hamNegative++;
                }
            }
            double overallA = ((double) (spamPositive + hamPositive) / (spamNegative + spamPositive + hamNegative + hamPositive));
            double sc = (double) spamPositive / (spamPositive + spamNegative);
            double bh = (double) hamNegative / (hamPositive + hamNegative);
            overall.add(overallA);
            spamsCaught.add(sc);
            blockedHams.add(bh);
        }

        System.out.println("overall accuracy : " + (overall.stream().reduce(Double::sum).get()) / runs);
        System.out.println("spams caught: " + (spamsCaught.stream().reduce(Double::sum).get()) / runs);
        System.out.println("hams blocked: " + (blockedHams.stream().reduce(Double::sum).get()) / runs);

    }

    private static DataSet stratification(LinkedList<String> spamSms, LinkedList<String> hamSms) {
        // apply knuth shuffle for uniform random distribution of samples
        Collections.shuffle(spamSms);
        Collections.shuffle(hamSms);
        // derive category percentage split
        double totalSamples = spamSms.size() + hamSms.size();
        double spamPercent = spamSms.size() / totalSamples;
        double hamPercent = hamSms.size() / totalSamples;

        // there should be 30% in validation set out of which 13% should be spam and 87 should be ham
        // stratification should be also applied to learning set and right now it is not
        int validationSize = (int) (0.3 * totalSamples);
        int hamValidationSize = (int) (validationSize * hamPercent);
        int spamValidationSize = (int) (validationSize * spamPercent);

        ArrayList<String> spamValidation = new ArrayList<>();
        for (int spam = 0; spam < spamValidationSize; spam++) {
            spamValidation.add(spamSms.remove(0));
        }

        ArrayList<String> hamValidation = new ArrayList<>();
        for (int ham = 0; ham < hamValidationSize; ham++) {
            hamValidation.add(hamSms.remove(0));
        }

        return new DataSet(spamSms, hamSms, spamValidation, hamValidation);
    }

    private static class DataSet {
        private List<String> trainingSpam;
        private List<String> trainingHam;
        private List<String> validationSpam;
        private List<String> validationHam;

        private DataSet(List<String> trainingSpam, List<String> trainingHam, List<String> validationSpam, List<String> validationHam) {
            this.trainingSpam = trainingSpam;
            this.trainingHam = trainingHam;
            this.validationSpam = validationSpam;
            this.validationHam = validationHam;
        }
    }
}
