package evaluation;

import algorithms.model.TextModel;
import algorithms.model.TextModelSupplier;
import evaluation.summary.Summary;
import evaluation.summary.SummaryAnalysis;
import utilities.ListUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class lets user evaluate {@link TextModel} and {@link algorithms.model.Model} implementations via train
 * and test strategy. The class returns various evaluation metrics wrapped in {@link Summary}, such as <em>overall accuracy</em>,
 * <em>per class accuracy</em> and <em>confusion matrix</em>.
 * User is able to specify number of iterations over which train and test will be averaged and validation ratio that shall
 * define percentage of all the samples which will be reserved for validation.
 *
 * <p>
 * The class will make <strong>stratified</strong> split of data since same validation ratio will be used for splits over each class. In other
 * words, validation set mimics classes distribution observed from samples.
 * </p>
 *
 * All the metrics are calculated over <strong>validation</strong> set only.
 *
 * @author dtemraz
 */
public class StratifiedTrainAndTest {

    /**
     * Returns {@link Summary} of the {@link TextModel} performance, given as average over <em>iterations</em> of train and test.
     * The summary includes overall accuracy, per class accuracy and confusion matrix.
     * <p>
     * The <em>modelSupplier</em> should return model given the training <em>data</em>, and data will be split into training
     * set and validation set, reserving the <em>validationRatio</em> of data for validation.
     * </p>
     * Data is split in stratified manner, validation set mimics class distribution observed from samples.
     *
     * @param modelSupplier supplies model which is trainable given the training <em>dat</em>
     * @param completeSet to train and evaluate model, split into training set and validation set according to validation ratio
     * @param validationRatio percentage of data reserved for validation
     * @param iterations over which summary results should be averaged
     * @return {@link Summary} averaged over number of <em>iterations</em>
     */
    public static Summary run(TextModelSupplier modelSupplier, Map<Double, List<String[]>> completeSet, double validationRatio, int iterations) {
        if (iterations <= 1) {
            TrainAndTestSplit<String[]> trainAndTestSplit = split(completeSet, validationRatio);
            return ModelEvaluation.execute(modelSupplier, trainAndTestSplit.trainingSet, trainAndTestSplit.validationSet);
        }
        // if there is more than one iteration, average results across them
        List<Summary> summaries = new ArrayList<>();
        for (int i = 0; i < iterations; i++) {
            TrainAndTestSplit<String[]> trainAndTestSplit = split(completeSet, validationRatio);
            summaries.add(ModelEvaluation.execute(modelSupplier, trainAndTestSplit.trainingSet, trainAndTestSplit.validationSet));
        }
        return SummaryAnalysis.average(summaries);
    }


    /**
     * Returns data sampled into training set and validation set, stratified distribution is maintained for each class in validation samples.
     *
     * @param data to split
     * @param validationRatio ratio of samples to reserve for validation
     * @param <T> type of data
     * @return data sampled into training set and validation set, stratified distribution is maintained for each class in validation samples
     */
    public static <T> TrainAndTestSplit<T> split(Map<Double, List<T>> data, double validationRatio) {
        HashMap<Double, List<T>> trainingSamples = new HashMap<>();
        HashMap<Double, List<T>> validationSamples = new HashMap<>();
        // iterate over samples for each class and split them into training set and validation set
        for (Map.Entry<Double, List<T>> entry : data.entrySet()) {
            List<List<T>> split = ListUtils.randomizedSplit(entry.getValue(), (1 - validationRatio));
            Double expectedClass = entry.getKey();
            trainingSamples.put(expectedClass, split.get(0));
            validationSamples.put(expectedClass, split.get(1));
        }
        return new TrainAndTestSplit<>(trainingSamples, validationSamples);
    }

}
