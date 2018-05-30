package evaluation;

import algorithms.ensemble.model.TextModelSupplier;
import structures.RandomizedQueue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class lets user evaluate {@link algorithms.ensemble.model.TextModel} performance with stratified k-fold validation.
 * The class returns various evaluation metrics wrapped in {@link Summary}, such as <em>overall accuracy</em>,
 * <em>per class accuracy</em> and <em>confusion matrix</em>.
 *
 * <p>All the metrics are calculated over <strong>validation</strong> set only.</p>
 *
 * The user may run k-fold in sequential mode with {@link #run(TextModelSupplier, Map, int)} or the parallel mode with
 * {@link #runParallel(TextModelSupplier, Map, int)}.
 *
 * <p> Note that parallel execution will require more ram, proportionally to number of folds. </p>
 *
 * @author dtemraz
 */
public class StratifiedKFold {

    /**
     * Returns {@link Summary} of the {@link algorithms.ensemble.model.TextModel} performance computed with stratified K-fold.
     * The summary includes overall accuracy, per class accuracy and confusion matrix.
     * <p>
     * The <em>modelSupplier</em> should return trained model given the training data. The <em>data</em> will be split into
     * k folds and each fold will be given a chance to be used once as a validation fold and k-1 times in training.
     * </p>
     *
     * <strong>Side-effect</strong> The method will attempt to reduce memory footprint and therefore during execution
     * will <em>release</en> memory occupied by lists in <em>data</em> map.
     *
     * @param modelSupplier supplies model which is trainable given the training <em>dat</em>
     * @param data to train and evaluate model, split into training set and validation set according to validation ratio
     * @param k number of folds
     * @return {@link Summary} averaged over each k-fold train/validate combination
     */
    public static Summary  run(TextModelSupplier modelSupplier, Map<Double, List<String[]>> data, int k) {
        return run(modelSupplier, getFolds(data, k));
    }

    /**
     * Returns {@link Summary} of the {@link algorithms.ensemble.model.TextModel} performance computed with stratified K-fold.
     * The summary includes overall accuracy, per class accuracy and confusion matrix.
     * <p>
     * The <em>modelSupplier</em> should return trained model given the training data. The <em>data</em> will be split into
     * k folds and each fold will be given a chance to be used once as a validation fold and k-1 times in training.
     * </p>
     *
     * This method will run each train/validation k-fold combination in parallel.
     * <p>
     * <strong>Note:</strong> for large data sets, this method will greatly increase consumed memory, proportional to number of folds,
     * and in absence of adequate RAM may in fact degrade performance compared to sequential mode.
     * </p>
     *
     * @param modelSupplier supplies model which is trainable given the training <em>dat</em>
     * @param data to train and evaluate model, split into training set and validation set according to validation ratio
     * @param k number of folds
     * @return {@link Summary} averaged over each k-fold train/validate combination
     */
    public static Summary runParallel(TextModelSupplier modelSupplier, Map<Double, List<String[]>> data, int k) {
        return runParallel(modelSupplier, getFolds(data, k));
    }

    // runs each of the train/validation k-fold combination in sequential mode
    private static Summary run(TextModelSupplier modelSupplier, ArrayList<Map<Double, List<String[]>>> folds) {
        List<Summary> summaries = new ArrayList<>();
        for (int run = 0, validationFold = 0; run < folds.size(); run++, validationFold++) {
            // at each iteration, use different fold for validation
            Map<Double, List<String[]>> trainingSet = extractTrainingSet(folds, validationFold);
            summaries.add(ModelEvaluation.execute(modelSupplier, extractTrainingSet(folds, validationFold), folds.get(validationFold)));
            trainingSet.replaceAll((k, v) -> new ArrayList<>());
        }
        return SummaryAnalysis.average(summaries);
    }

    // returns training folds as a single map, skipping the fold with validationFold index
    private static Map<Double, List<String[]>> extractTrainingSet(ArrayList<Map<Double, List<String[]>>> folds, int validationFold) {
        Map<Double, List<String[]>> trainingSet = new HashMap<>();
        folds.get(0).keySet().forEach(classId -> trainingSet.put(classId, new ArrayList<>()));
        for (int fold = 0; fold < folds.size(); fold++) {
            if (fold == validationFold) {
                continue;
            }
            folds.get(fold).forEach((k, v) -> trainingSet.get(k).addAll(v));
        }
        return trainingSet;
    }

    // runs each of the train/validation k-fold combination in parallel mode, requires considerably more ram than sequential mode
    private static Summary runParallel(TextModelSupplier modelSupplier, ArrayList<Map<Double, List<String[]>>> folds) {
        // reclaim thread pool resources since this will be very infrequent action
        ExecutorService executorService = Executors.newWorkStealingPool();
        List<Summary> summaries = new ArrayList<>();
        try {
            for (Future<Summary> future : executorService.invokeAll(prepareParallelExecutions(modelSupplier, folds))) {
                summaries.add(future.get());
            }
        } catch (Exception e) {
            throw new IllegalStateException("failed parallel execution of k-fold validation", e);
        }
        finally {
            executorService.shutdownNow();
        }
        return SummaryAnalysis.average(summaries);
    }

    // creates callable instances for model execution per each k fold combination
    private static List<Callable<Summary>> prepareParallelExecutions(TextModelSupplier modelSupplier, ArrayList<Map<Double, List<String[]>>> folds) {
        List<Callable<Summary>> modelExecutions = new ArrayList<>();
        int validationFold = 0;
        for (int run = 0; run < folds.size(); run++, validationFold++) {
            modelExecutions.add(new FoldsExecutor(modelSupplier, folds, validationFold));
        }
        return modelExecutions;
    }

    // returns data sampled into k folds, where each fold contains same number of elements per class
    private static ArrayList<Map<Double, List<String[]>>> getFolds(Map<Double, List<String[]>> data, int k) {
        ArrayList<Map<Double, List<String[]>>> folds = new ArrayList<>();
        // initialize empty k folds with only class ids
        for (int i = 0; i < k; i++) {
            Map<Double, List<String[]>> fold = new HashMap<>();
            data.keySet().forEach(classId -> fold.put(classId, new ArrayList<>()));
            folds.add(fold);
        }
        // divide data for each class into k folds
        for (Map.Entry<Double, List<String[]>> classData : data.entrySet()) {
            List<String[]> samples = classData.getValue();
            // divide samples from a single class into k sub samples
            ArrayList<ArrayList<String[]>> kFolds = divide(samples, k);
            // add class sub samples to existing folds
            for (int fold = 0; fold < kFolds.size(); fold++) {
                folds.get(fold).get(classData.getKey()).addAll(kFolds.get(fold));
            }
        }
        // saving memory here since these references over huge data sets inccur considerable memory cost
        data.replaceAll((classId, samples) -> new ArrayList<>());

        return folds;
    }

    /**
     * Splits the <em>data</em> list into <em>k</em> sub-lists, each  having a same number of elements. Remaining elements,
     * up to k-1, are discarded.
     *
     * @param data to split into <em>k</em> sub-lists
     * @param k number of sub-lists to split <em>data</em> into
     * @return data split into <em>k</em> sub-lists
     */
    private static ArrayList<ArrayList<String[]>> divide(List<String[]> data, int k) {
        ArrayList<ArrayList<String[]>> samples = new ArrayList<>();
        int foldSize = (int) ((1D / k) * data.size());
        // initialize randomized queue with indexes of possible elements in data array list
        RandomizedQueue<Integer> indexes = RandomizedQueue.intQueue(data.size());
        int foldElements = 0;
        // up to %k(0-9 usually) elements might end up being discarded if the division was not perfect
        for (int i = 0; i < k; i++, foldElements = 0) {
            ArrayList<String[]> fold = new ArrayList<>();
            // pick random elements from data until a fold is full, then move to a next fold
            while (foldElements++ < foldSize) {
                fold.add(data.get(indexes.dequeue()));
            }
            samples.add(fold);
        }
        return samples;
    }

    /**
     * This class is there to enable parallel execution of k-fold combinations. The lambda expressions aren't used to create
     * {@link Callable} implementations since it's annoying to pass the different <em>validationFold</em> in a loop to each
     * instance due to java rules for effectively final variables.
     */
    private static class FoldsExecutor implements Callable<Summary> {

        private final TextModelSupplier modelSupplier;
        private final ArrayList<Map<Double, List<String[]>>> folds;
        private final int validationFold;

        private FoldsExecutor(TextModelSupplier modelSupplier, ArrayList<Map<Double, List<String[]>>> folds, int validationFold) {
            this.modelSupplier = modelSupplier;
            this.folds = folds;
            this.validationFold = validationFold;
        }

        @Override
        public Summary call() {
            Map<Double, List<String[]>> trainingSet = extractTrainingSet(folds, validationFold);
            Map<Double, List<String[]>> validationSet = folds.get(validationFold);
            return ModelEvaluation.execute(modelSupplier, trainingSet, validationSet);
        }
    }
}
