package evaluation;

import algorithms.model.TextModel;
import algorithms.model.TextModelSupplier;
import evaluation.summary.Summary;
import evaluation.summary.WronglyClassified;
import lombok.RequiredArgsConstructor;

import java.util.*;

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
     * @param trainingSet to train the model to evaluate
     * @param validationSet to validate performance of the model
     * @return report of a model evaluation trained with <em>trainingSet</em> and validated with <em>validationSet</em>
     */
    public static Summary execute(TextModelSupplier modelSupplier, Map<Double, List<String[]>> trainingSet, Map<Double, List<String[]>> validationSet) {
        TextModel textModel = modelSupplier.get(trainingSet);
        Evaluation evaluation = prepareEvaluation(validationSet.keySet());
        for (Map.Entry<Double, List<String[]>> validationSamples : validationSet.entrySet()) {
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (String[] sample : validationSamples.getValue()) {
                double predicted = textModel.classify(sample);
                if (predicted != expectedClass) {
                    evaluation.wronglyClassified.add(new WronglyClassified(expectedClass, predicted, removeLatentFeatures(sample)));
                }
                evaluation.confusionMatrix.get(expectedClass).merge(predicted, 1, Integer::sum);
            }
        }
        return new Summary(evaluation.confusionMatrix, evaluation.wronglyClassified);
    }

    /**
     * Returns {@link Summary} report from evaluation of a trained <em>textModel</em> over <em>validationSet</em>.
     *
     * @param textModel already trained model
     * @param validationSet to validate performance of the model
     * @return report of <em>textModel</em> evaluation validated with <em>validationSet</em>
     */
    public static Summary execute(TextModel textModel, Map<Double, List<IdentifiableSample>> validationSet) {
        Evaluation evaluation = prepareEvaluation(validationSet.keySet());
        for (Map.Entry<Double, List<IdentifiableSample>> validationSamples : validationSet.entrySet()) {
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (IdentifiableSample sample : validationSamples.getValue()) {
                String[] features = sample.getFeatures();
                double predicted = textModel.classify(sample.getFeatures());
                if (predicted != expectedClass) {
                    evaluation.wronglyClassified.add(new WronglyClassified(expectedClass, predicted, removeLatentFeatures(features), sample.getId()));
                }
                evaluation.confusionMatrix.get(expectedClass).merge(predicted, 1, Integer::sum);
            }
        }
        return new Summary(evaluation.confusionMatrix, evaluation.wronglyClassified);
    }

    // use Summary instance as a holder object for confusion matrix and wrongly classified messages
    private static Evaluation prepareEvaluation(Set<Double> classes) {
        HashSet<WronglyClassified> wronglyClassified = new HashSet<>();
        Map<Double, Map<Double, Integer>> confusionMatrix = new HashMap<>();
        classes.forEach((classId) -> {
            HashMap<Double, Integer> classPredictions = new HashMap<>();
            classes.forEach(c -> classPredictions.put(c, 0));
            confusionMatrix.put(classId, classPredictions);
        });
        return new Evaluation(confusionMatrix, wronglyClassified);
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

    // a tiny wrapper class to hold confusion matrix and a set of wrongly classified messages
    @RequiredArgsConstructor
    private static class Evaluation {
        private final Map<Double, Map<Double, Integer>> confusionMatrix;
        private final HashSet<WronglyClassified> wronglyClassified;
    }

}
