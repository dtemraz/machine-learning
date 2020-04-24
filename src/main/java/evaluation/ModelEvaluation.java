package evaluation;

import algorithms.model.TextModel;
import algorithms.model.TextModelSupplier;
import evaluation.summary.Summary;
import evaluation.summary.WronglyClassified;
import lombok.RequiredArgsConstructor;

import java.util.*;

/**
 * This class lets the user obtain {@link Summary} report from the evaluation of {@link TextModel}. There is only one method
 * in the class{@link #trainAndValidate(TextModelSupplier, Map, Map)} and it's used to train the model with training set data and validate
 * model performance with validation set.
 *
 * @author dtemraz
 */
public class ModelEvaluation {

    /**
     * Returns {@link Summary} report from evaluation of a model defined with <em>modelSupplier</em>. The model is trained with
     * <em>trainingSet</em> and the performance is validated with <em>validationSet</em>.
     *
     * @param modelSupplier to produce trained model, given the <em>trainingSet</em>
     * @param trainingSet to train the model to evaluate
     * @param validationSet to validate performance of the model
     * @return report of a model evaluation trained with <em>trainingSet</em> and validated with <em>validationSet</em>
     */
    public static Summary trainAndValidate(TextModelSupplier modelSupplier, Map<Double, List<String[]>> trainingSet, Map<Double, List<String[]>> validationSet) {
        return validate(modelSupplier.get(trainingSet), validationSet);
    }

    /**
     * Returns {@link Summary} report from validation of a trained <em>textModel</em> over <em>validationSet</em>.
     *
     * @param textModel already trained model
     * @param validationSet to validate performance of the model
     * @return report of <em>textModel</em> evaluation validated with <em>validationSet</em>
     */
    public static Summary validate(TextModel textModel, Map<Double, List<String[]>> validationSet) {
        Evaluation evaluation = prepareEvaluation(validationSet.keySet());
        for (Map.Entry<Double, List<String[]>> validationSamples : validationSet.entrySet()) {
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (String[] sample : validationSamples.getValue()) {
                double predicted = textModel.classify(sample);
                if (predicted != expectedClass) {
                    evaluation.wronglyClassified.add(new WronglyClassified(expectedClass, predicted, String.join(" ", sample)));
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
    public static Summary validateWithIdSamples(TextModel textModel, Map<Double, List<IdentifiableSample>> validationSet) {
        Evaluation evaluation = prepareEvaluation(validationSet.keySet());
        for (Map.Entry<Double, List<IdentifiableSample>> validationSamples : validationSet.entrySet()) {
            double expectedClass = validationSamples.getKey();
            // iterate all validation samples of a given class
            for (IdentifiableSample sample : validationSamples.getValue()) {
                String[] features = sample.getFeatures();
                double predicted = textModel.classify(sample.getFeatures());
                if (predicted != expectedClass) {
                    evaluation.wronglyClassified.add(new WronglyClassified(expectedClass, predicted, String.join(" ", features), sample.getId()));
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


    // a tiny wrapper class to hold confusion matrix and a set of wrongly classified messages
    @RequiredArgsConstructor
    private static class Evaluation {
        private final Map<Double, Map<Double, Integer>> confusionMatrix;
        private final HashSet<WronglyClassified> wronglyClassified;
    }

}