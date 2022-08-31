package examples;

import algorithms.linear_regression.SoftMaxRegression;
import algorithms.linear_regression.optimization.multiclass.MultiClassOptimizer;
import algorithms.linear_regression.optimization.multiclass.SoftMaxOptimizer;
import algorithms.model.TextModel;
import evaluation.ModelEvaluation;
import evaluation.StratifiedTrainAndTest;
import evaluation.TrainAndTestSplit;
import evaluation.summary.Summary;
import structures.text.Vocabulary;
import textEmbedding.PoolingWithEmbedding;
import textEmbedding.embeddings.WeightedWordEmbeddings;
import textEmbedding.embeddings.WordEmbeddings;
import textEmbedding.pooling.Pooling;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public class TestBankingData {
    private static final Pattern WS = Pattern.compile(" ");
    private static final Predicate<Character> IS_WHITESPACE = Character::isWhitespace;
    private static final Predicate<Character> IS_LETTER_OR_DIGIT = Character::isLetterOrDigit;

    private static final Map<String, Double> classLabel = Map.of("CHECK_ACCOUNT_BALANCE", 0D,
                                                                 "LOST_OR_STOLEN_CARD", 1D,
                                                                 "TRANSFER_MONEY", 2D,
                                                                 "GO_TO_AGENT", 3D,
                                                                 "VIEW_TRANSACTIONS", 4D,
                                                                 "NEAREST_BRANCH_OR_ATM", 5D,
                                                                 "GET_EXCHANGE_RATES", 6D,
                                                                 "CREDIT_INFORMATION", 7D);

    public static void main(String[] args) throws Exception {
        Map<Double, List<String[]>> data = loadData();
        TrainAndTestSplit<String[]> trainAndTestSplit = StratifiedTrainAndTest.split(data, 0.2);
        HashMap<Double, List<String[]>> trainingSet = trainAndTestSplit.getTrainingSet();
        HashMap<Double, List<String[]>> validationSet = trainAndTestSplit.getValidationSet();

        Vocabulary v = new Vocabulary(trainingSet.values());
        MultiClassOptimizer softMaxOptimizer = new SoftMaxOptimizer(0.2, 100);

        // WeightedWordEmbeddings weightedWordEmbeddings = new WeightedWordEmbeddings(v, WordEmbeddings.load("/Users/dtemraz/Downloads/crawl-clean#300d.vec"));
        // PoolingWithEmbedding poolingWithEmbedding = new PoolingWithEmbedding(Pooling.AVG, weightedWordEmbeddings);
        TextModel model = SoftMaxRegression.getTextModel(v, trainingSet, softMaxOptimizer);
        // TextModel model = SoftMaxRegression.getWordEmbeddingsModel(trainingSet, poolingWithEmbedding, softMaxOptimizer);

        //MultinomialNaiveBayes model = new MultinomialNaiveBayes(trainingSet);
        Summary validate = ModelEvaluation.validate(model, validationSet);
        System.out.println(validate);

    }

 /*   private static TextModel embeddingsModel(Vocabulary v, HashMap<Double, List<String[]>> trainingSet) {
        WeightedWordEmbeddings weightedWordEmbeddings = new WeightedWordEmbeddings(v, WordEmbeddings.load("/Users/dtemraz/Downloads/crawl-clean#300d.vec"));
        PoolingWithEmbedding poolingWithEmbedding = new PoolingWithEmbedding(Pooling.AVG, weightedWordEmbeddings);
        return SoftMaxRegression.getTextModel(v, trainingSet, softMax);
    }*/

    private static Map<Double, List<String[]>> loadData() throws IOException {
        List<String> lines = Files.readAllLines(Path.of("/Users/dtemraz/IdeaProjects/science/machine-learning/src/test/java/examples/appen_data_set.csv"));
        Map<Double, List<String[]>> dataset = new HashMap<>();
        classLabel.keySet().forEach(l -> dataset.put(classLabel.get(l), new ArrayList<>()));
        for (String line : lines) {
            String[] sampleLabel = line.split(";");
            String label = sampleLabel[1];
            Double classId = classLabel.get(label);
            if (classId == null) {
                throw new IllegalArgumentException("null class id: " + label);
            }
            String[] words = WS.split(clean(sampleLabel[0]));
            dataset.get(classId).add(words);
        }
        return dataset;
    }

    private static String clean(String text) {
        return keepLettersDigitsAndWhitespaces(text.trim().toLowerCase());
    }

    public static String keepLettersDigitsAndWhitespaces(String text) {
        return filterCharacters(text, IS_WHITESPACE.or(IS_LETTER_OR_DIGIT));
    }

    private static String filterCharacters(String text, Predicate<Character> filter) {
        if (text == null || text.isEmpty()) {
            return "";
        }

        int length = text.length();
        char[] cleanedText = new char[length];
        int cleanedTextLength = 0;

        for (int i = 0; i < length; i++) {
            char c = text.charAt(i);

            if (filter.test(c)) {
                cleanedText[cleanedTextLength] = c;
                cleanedTextLength++;
            }
        }

        return new String(cleanedText, 0, cleanedTextLength);
    }

}
