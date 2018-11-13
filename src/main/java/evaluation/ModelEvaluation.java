package evaluation;

import algorithms.model.TextModel;
import algorithms.model.TextModelSupplier;
import evaluation.summary.Summary;
import evaluation.summary.WronglyClassified;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * This class lets the user obtain {@link Summary} report from the evaluation of {@link TextModel}. There is only one method
 * in the class{@link #execute(TextModelSupplier, Map, Map)} and it's used to train the model with training set data and validate
 * model performance with validation set.
 *
 * @author dtemraz
 */
public class ModelEvaluation {

    private static final String LATENT_FEATURE_DELIMITER = "_x_";

    /**
     * Returns {@link Summary} report from evaluation of a model defined with <em>modelSupplier</em>. The model is trained with
     * <em>trainingSet</em> and the performance is validated with <em>validationSet</em>.
     *
     * @param modelSupplier to produce trained model, given the <em>trainingSet</em>
     * @param trainingSet   to train the model to evaluate
     * @param validationSet to validate performance of the model
     * @return report of a model evaluation trained with <em>trainingSet</em> and validated with <em>validationSet</em>
     */
    public static Summary execute(TextModelSupplier modelSupplier, Map<Double, List<String[]>> trainingSet, Map<Double, List<String[]>> validationSet) {
        TextModel textModel = modelSupplier.get(trainingSet);
        Summary s = prepareSummaryMetrics(validationSet.keySet());
        // TODO consider to implemented this with method chaining, although additional memory will be required for converted validation set, not ideal if id is not important
        double correct = 0;
        for (Map.Entry<Double, List<String[]>> validationSamples : validationSet.entrySet()) {
            int positive = 0; // per class
            int negative = 0; // per class
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (String[] sample : validationSamples.getValue()) {
                Double predicted = textModel.classify(sample);
                if (predicted == expectedClass) {
                    positive++;
                    correct++;
                } else {
                    negative++;
                    s.getWronglyClassified().add(new WronglyClassified(expectedClass, predicted, removeLatentFeatures(sample)));
                }
                s.getConfusionMatrix().get(expectedClass).merge(predicted, 1, (old, n) -> old + n);
            }
            s.getClassAccuracy().put(expectedClass, (positive / (double) (positive + negative)));
        }
        // ratio of all correctly classified samples divided by a total number of samples
        double overallAccuracy = correct / (validationSet.entrySet().stream().map((e) -> e.getValue().size()).reduce(Integer::sum)).get();
        return new Summary(overallAccuracy, s.getClassAccuracy(), s.getConfusionMatrix(), s.getWronglyClassified());
    }

    /**
     * Returns {@link Summary} report from evaluation of a trained <em>textModel</em> over <em>validationSet</em>.
     *
     * @param textModel already trained model
     * @param validationSet to validate performance of the model
     * @return report of <em>textModel</em> evaluation validated with <em>validationSet</em>
     */
    public static Summary execute(TextModel textModel, Map<Double, List<IdentifiableSample>> validationSet) {
        Summary s = prepareSummaryMetrics(validationSet.keySet());
        // correct predictions across all classes
        double correct = 0;
        for (Map.Entry<Double, List<IdentifiableSample>> validationSamples : validationSet.entrySet()) {
            int positive = 0; // per class
            int negative = 0; // per class
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (IdentifiableSample sample : validationSamples.getValue()) {
                String[] features = sample.getFeatures();
                Double predicted = textModel.classify(sample.getFeatures());
                if (predicted == expectedClass) {
                    positive++;
                    correct++;
                } else {
                    negative++;
                    s.getWronglyClassified().add(new WronglyClassified(expectedClass, predicted, removeLatentFeatures(features), sample.getId()));
                }
                s.getConfusionMatrix().get(expectedClass).merge(predicted, 1, (old, n) -> old + n);
            }
            s.getClassAccuracy().put(expectedClass, (positive / (double) (positive + negative)));
        }
        // ratio of all correctly classified samples divided by a total number of samples
        double overallAccuracy = correct / (validationSet.entrySet().stream().map((e) -> e.getValue().size()).reduce(Integer::sum)).get();
        return new Summary(overallAccuracy, s.getClassAccuracy(), s.getConfusionMatrix(), s.getWronglyClassified());
    }

    private static Summary prepareSummaryMetrics(Set<Double> classes) {
        HashSet<WronglyClassified> wronglyClassified = new HashSet<>();
        // percentage of correct samples over all samples per class
        HashMap<Double, Double> classAccuracy = new HashMap<>();
        // how many times a class was mistaken by another class
        Map<Double, Map<Double, Integer>> confusionMatrix = new HashMap<>();
        // put all expected classes in confusion matrix as keys
        classes.forEach((classId) -> confusionMatrix.put(classId, new HashMap<>()));
        // correct predictions across all classes
        return new Summary(0, classAccuracy, confusionMatrix, wronglyClassified);
    }

    private static String removeLatentFeatures(String[] features) {
        StringJoiner joiner = new StringJoiner(" ");
        for (String feature : features) {
            // latent features are grouped together and put at the end, so we can stop on first
            if (feature.startsWith(LATENT_FEATURE_DELIMITER)) {
                break;
            }
            joiner.add(feature);
        }
        return joiner.toString();
    }

}
