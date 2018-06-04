package evaluation;

import algorithms.ensemble.model.TextModel;
import algorithms.ensemble.model.TextModelSupplier;
import evaluation.summary.Summary;
import evaluation.summary.WronglyClassified;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * This class lets the user obtain {@link Summary} report from the evaluation of {@link TextModel}. There is only one method
 * in the class{@link #execute(TextModelSupplier, Map, Map)} and it's used to train the model with training set data and validate
 * model performance with validation set.
 *
 * @author dtemraz
 */
class ModelEvaluation {

    private static final String LATENT_FEATURE_DELIMITER = "_x_";

    /**
     * Returns {@link Summary} report from evaluation of a model defined with <em>modelSupplier</em>. The model is trained with
     * <em>trainingSet</em> and the performance is validated with <em>validationSet</em>.
     *
     * @param modelSupplier to produce trained model, given the <em>trainingSet</em>
     * @param trainingSet to train the model to evaluate
     * @param validationSet to validate performance of the model
     * @return report of a model evaluation trained with <em>trainingSet</em> and validated with <em>validationSet</em>
     */
    static Summary execute(TextModelSupplier modelSupplier, Map<Double, List<String[]>> trainingSet, Map<Double, List<String[]>> validationSet) {
        HashSet<WronglyClassified> wronglyClassified = new HashSet<>();
        // percentage of correct samples over all samples per class
        HashMap<Double, Double> classAccuracy = new HashMap<>();
        // how many times a class was mistaken by another class
        Map<Double, Map<Double, Integer>> confusionMatrix = new HashMap<>();
        // put all expected classes in confusion matrix as keys
        trainingSet.keySet().forEach((classId) -> confusionMatrix.put(classId, new HashMap<>()));
        // correct predictions across all classes
        double correct = 0;

        // train the model with the training set extracted from data
        TextModel model = modelSupplier.get(trainingSet);

        for (Map.Entry<Double, List<String[]>> validationSamples : validationSet.entrySet()) {
            int positive = 0; // per class
            int negative = 0; // per class
            double expectedClass = validationSamples.getKey();

            // iterate all validation samples of a given class
            for (String[] sample : validationSamples.getValue()) {
                Double predicted = model.classify(sample);
                if (predicted == expectedClass) {
                    positive++;
                    correct++;
                } else {
                    negative++;
                    wronglyClassified.add(new WronglyClassified(expectedClass, predicted, removeLatentFeatures(sample)));
                }
                confusionMatrix.get(expectedClass).merge(predicted, 1, (old, n) -> old + n);
            }
            classAccuracy.put(expectedClass, (positive / (double) (positive + negative)));
        }
        // ratio of all correctly classified samples divided by a total number of samples
        double overallAccuracy = correct / (validationSet.entrySet().stream().map((e) -> e.getValue().size()).reduce(Integer::sum)).get();
        return new Summary(overallAccuracy, classAccuracy, confusionMatrix, wronglyClassified);
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
