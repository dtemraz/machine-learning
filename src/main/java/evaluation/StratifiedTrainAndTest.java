package evaluation;

import algorithms.ensemble.model.TextModel;
import algorithms.ensemble.model.TextModelSupplier;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This class lets user evaluate {@link TextModel} and {@link algorithms.ensemble.model.Model} implementations via train
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
     * @param data to train and evaluate model, split into training set and validation set according to validation ratio
     * @param validationRatio percentage of data reserved for validation
     * @param iterations over which summary results should be averaged
     * @return {@link Summary} averaged over number of <em>iterations</em>
     */
    public static Summary run(TextModelSupplier modelSupplier, Map<Double, List<String[]>> data, double validationRatio, int iterations) {
        if (iterations <= 1) {
            TrainAndTestSplit<String[]> trainAndTestSplit = split(data, validationRatio);
            return ModelEvaluation.execute(modelSupplier, trainAndTestSplit.trainingSet, trainAndTestSplit.validationSet);
        }

        // if there is more than one iteration, average results across them
        List<Summary> summaries = new ArrayList<>();
        for (int i = 0; i < iterations; i++) {
            TrainAndTestSplit<String[]> trainAndTestSplit = split(data, validationRatio);
            summaries.add(ModelEvaluation.execute(modelSupplier, trainAndTestSplit.trainingSet, trainAndTestSplit.validationSet));
        }
        return SummaryAnalysis.average(summaries);
    }

    // returns data sampled into training set and validation set, for all classes validation ratio samples is reserved for validation
    private static TrainAndTestSplit<String[]> split(Map<Double, List<String[]>> data, double validationRatio) {
        // populate training and validation maps with class ids and associated empty lists of samples
        Set<Double> classIds = data.keySet();
        HashMap<Double, List<String[]>> trainingSamples = new HashMap<>();
        classIds.forEach(classId -> trainingSamples.put(classId, new ArrayList<>()));
        HashMap<Double, List<String[]>> validationSamples = new HashMap<>();
        classIds.forEach(classId -> validationSamples.put(classId, new ArrayList<>()));

        // iterate over samples for each class and split them into training set and validation set
        for (Map.Entry<Double, List<String[]>> entry : data.entrySet()) {
            List<String[]> classTexts = entry.getValue();
            Double expectedClass = entry.getKey();
            ListSplit<String[]> split = split(classTexts, validationRatio);
            trainingSamples.put(expectedClass, split.trainingSet);
            validationSamples.put(expectedClass, split.validationSet);
        }

        return new TrainAndTestSplit<>(trainingSamples, validationSamples);
    }

    /**
     * Returns <em>samples</em> split into training set and validation set, with <em>validationRatio</em> samples reserved for validation.
     *
     * @param samples to split into validation and training set
     * @param validationRatio percentage of all samples reserved for validation set
     * @param <T> type of sample
     * @return <em>samples</em> split into training set and validation set
     */
    private static <T> ListSplit<T> split(List<T> samples, double validationRatio) {
        ArrayList<T> validation = new ArrayList<>();
        ArrayList<T> training = new ArrayList<>();
        samples.forEach(sample -> {
            if (Math.random() < validationRatio) {
                validation.add(sample);
            } else
                training.add(sample);
        });
        return new ListSplit<>(training, validation);
    }

    /**
     * Simple wrapper class used to contain both the training set and validation set derived from data.
     *
     * @param <T> type of data in the sets
     */
    @RequiredArgsConstructor
    private static class ListSplit<T> {
        private final List<T> trainingSet;
        private final List<T> validationSet;
    }

    /**
     * Simple wrapper class used to contain both the training set and validation set across all classes.
     *
     * @param <T> type of data in the sets
     */
    @RequiredArgsConstructor
    private static class TrainAndTestSplit<T> {
        private final HashMap<Double, List<T>> trainingSet;
        private final HashMap<Double, List<T>> validationSet;
    }

}
