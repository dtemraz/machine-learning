package examples.sms_spam;

import algorithms.bayes.MultinomialNaiveBayes;
import algorithms.linear_regression.optimization.multiclass.MultiClassOptimizer;
import algorithms.linear_regression.optimization.multiclass.ParallelSoftMaxOptimizer;
import algorithms.linear_regression.optimization.multiclass.SoftMaxOptimizer;
import algorithms.linear_regression.optimization.text.*;
import structures.text.Vocabulary;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author dtemraz
 */
public class TestSmsSpam {

    static int runs = 1;

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

            Map<Double, List<String[]>> smsCorpus = new HashMap<>();
            smsCorpus.put(0D, dataSet.trainingSpam.stream().map(String::toLowerCase).map(t -> t.trim().split("\\s+")).collect(Collectors.toList()));
            smsCorpus.put(1D, dataSet.trainingHam.stream().map(String::toLowerCase).map(t -> t.trim().split("\\s+")).collect(Collectors.toList()));

            ArrayList<String[]> combined = new ArrayList<>(smsCorpus.get(0D));
            combined.addAll(smsCorpus.get(1D));
            Vocabulary v = new Vocabulary(combined);

            SquaredErrorStoppingCriteria stoppingCriteria = SquaredErrorStoppingCriteria.squaredErrorBellowTolerance(0.004);

            TextOptimizer sgd = (x, w) -> new TextGradientDescent(0.0003, 10_000, stoppingCriteria, 0, true).stochastic(x, w, v);
            TextOptimizer optimizerBatch = (x, w) -> new TextGradientDescent(0.003, 10_000, stoppingCriteria, 0.1, true).miniBatch(x, w, 120, v);
            TextOptimizer hogwild = (x, w) -> new ParallelTextGradientDescent(0.0003, 10_000, stoppingCriteria, 0, true).stochastic(x, w, v);
            MultiClassOptimizer softMax = new SoftMaxOptimizer(0.0003, 10_000, 0);
            MultiClassOptimizer parallelSoftMax = new ParallelSoftMaxOptimizer(0.0003, 10_000, 0, true);

//            epoch: 9999 , squared error: 11.175443484192442
//            converged in: 10000 epochs, epoch error: 11,175443
//            training time: 12
//            overall accuracy : 0.9856373429084381
//            spams caught: 0.9196428571428571
//            hams blocked: 0.00414651002073255

//            TextModel textModel = OneAgainstRest.getTextModel(v, smsCorpus, hogwild);
//            TextModel textModel = LogisticRegression.getTextModel(v, smsCorpus, hogwild);
//            TextModel textModel = WithThreshold.textModel(LogisticRegression.getTextModel(v, smsCorpus, optimizerBatch), 0.5);
            // TextModel textModel = SoftMaxRegression.getTextModel(v, smsCorpus, parallelSoftMax);

            MultinomialNaiveBayes textModel = new MultinomialNaiveBayes(smsCorpus);

            // TextModelWithProbability textModel = SoftMaxRegression.getTextModelWithProbabilities(v, smsCorpus, parallelSoftMax);
            // TextModel textModel = SoftMaxRegression.getWordEmbeddingsModel(smsCorpus);

            int spamPositive = 0;
            int spamNegative = 0;
            for (String message : dataSet.validationSpam) {
                // ClassificationResult classificationResult = textModelWithProbabilities.classifyWithProb(message.trim().toLowerCase().split("\\s+"));
                // System.out.println(classificationResult.getPredictions().values().stream().reduce(0D, Double::sum).doubleValue());
//                if (oneVSrest.classify(message.trim().toLowerCase().split("\\s+")) == 0D) {
                if (textModel.classify(message.trim().toLowerCase().split("\\s+")) < 0.5) {
                    spamPositive++;
                } else {
                    spamNegative++;
                }
            }

            int hamPositive = 0;
            int hamNegative = 0;
            for (String message : dataSet.validationHam) {
//                if (oneVSrest.classify(message.trim().toLowerCase().split("\\s+")) == 1D) {
                if (textModel.classify(message.trim().toLowerCase().split("\\s+")) >= 0.5D) {
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
