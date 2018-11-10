// TODO - Composite (Linked) list implementation for constant time view of merged lists as a single list

//package evaluation;

//import algorithms.model.TextModelSupplier;
//import evaluation.summary.Summary;
//import evaluation.summary.SummaryAnalysis;
//import utilities.ListUtils;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.concurrent.Callable;
//import java.util.concurrent.ExecutorService;
//import java.util.concurrent.Executors;
//import java.util.concurrent.Future;
//
///**
// * This class lets user evaluate {@link algorithms.model.TextModel} performance with stratified k-fold validation.
// * The class returns various evaluation metrics wrapped in {@link Summary}, such as <em>overall accuracy</em>,
// * <em>per class accuracy</em> and <em>confusion matrix</em>.
// * <p>
// * <p>All the metrics are calculated over <strong>validation</strong> set only.</p>
// * <p>
// * The user may run k-fold in sequential mode with {@link #run(TextModelSupplier, Map, int)} or the parallel mode with
// * {@link #runParallel(TextModelSupplier, Map, int)}.
// * <p>
// * <p> Note that parallel execution will require more ram, proportionally to number of folds. </p>
// *
// * @author dtemraz
// */
//
//public class KFold {
//
//    /**
//     * Returns {@link Summary} of the {@link algorithms.model.TextModel} performance computed with stratified K-fold.
//     * The summary includes overall accuracy, per class accuracy and confusion matrix.
//     * <p>
//     * The <em>modelSupplier</em> should return trained model given the training data. The <em>data</em> will be split into
//     * k folds and each fold will be given a chance to be used once as a validation fold and k-1 times in training.
//     * </p>
//     * <p>
//     * <strong>Side-effect</strong> The method will attempt to reduce memory footprint and therefore during execution
//     * will <em>release</en> memory occupied by lists in <em>data</em> map.
//     *
//     * @param modelSupplier supplies model which is trainable given the training <em>dat</em>
//     * @param data          to train and evaluate model, split into training set and validation set according to validation ratio
//     * @param k             number of folds
//     * @return {@link Summary} averaged over each k-fold train/validate combination
//     */
//    public static Summary run(TextModelSupplier modelSupplier, Map<Double, List<String[]>> data, int k) {
//        return run(modelSupplier, partition(data, k));
//    }
//
//
//    private static List<Map<Double, List<String[]>>> partition(Map<Double, List<String[]>> data, int k) {
//        List<Map<Double, List<String[]>>> kFolds = new ArrayList<>();
//        // prepare empty k folds, each map has unique partition of samples for each class
//        for (int i = 0; i < k; i++) {
//            kFolds.add(new HashMap<>());
//        }
//        // partitions samples for each class into k equally sized partitions which are assigned to different folds
//        for (Map.Entry<Double, List<String[]>> entry : data.entrySet()) {
//            Double classId = entry.getKey();
//            // partition list into k partitions, each partition reserved for a single fold
//            List<List<String[]>> partitions = ListUtils.randomizedEqualPartitions(entry.getValue(), k);
//            // assign each partition to different fold for this classId
//            for (int i = 0; i < partitions.size(); i++) {
//                kFolds.get(i).put(classId, partitions.get(i));
//            }
//        }
//        return kFolds;
//    }
//
//    /**
//     * Returns {@link Summary} of the {@link algorithms.model.TextModel} performance computed with stratified K-fold.
//     * The summary includes overall accuracy, per class accuracy and confusion matrix.
//     * <p>
//     * The <em>modelSupplier</em> should return trained model given the training data. The <em>data</em> will be split into
//     * k folds and each fold will be given a chance to be used once as a validation fold and k-1 times in training.
//     * </p>
//     * <p>
//     * This method will run each train/validation k-fold combination in parallel.
//     * <p>
//     * <strong>Note:</strong> for large data sets, this method will greatly increase consumed memory, proportional to number of folds,
//     * and in absence of adequate RAM may in fact degrade performance compared to sequential mode.
//     * </p>
//     *
//     * @param modelSupplier supplies model which is trainable given the training <em>dat</em>
//     * @param data          to train and evaluate model, split into training set and validation set according to validation ratio
//     * @param k             number of folds
//     * @return {@link Summary} averaged over each k-fold train/validate combination
//     */
//    public static Summary runParallel(TextModelSupplier modelSupplier, Map<Double, List<String[]>> data, int k) {
//        return runParallel(modelSupplier, partition(data, k));
//    }
//
//    // runs each of the train/validation k-fold combination in sequential mode
//    private static Summary run(TextModelSupplier modelSupplier, List<Map<Double, List<String[]>>> folds) {
//        List<Summary> summaries = new ArrayList<>();
//        for (int run = 0, validationFold = 0; run < folds.size(); run++, validationFold++) {
//            // at each iteration, use different fold for validation
//            summaries.add(ModelEvaluation.execute(modelSupplier, extractTrainingSet(folds, validationFold), folds.get(validationFold)));
//        }
//        return SummaryAnalysis.average(summaries);
//    }
//
//    // returns training folds as a single map, skipping the fold with validationFold index
//    private static Map<Double, List<String[]>> extractTrainingSet(List<Map<Double, List<String[]>>> folds, int validationFold) {
//        Map<Double, List<String[]>> trainingSet = new HashMap<>();
//        folds.get(0).keySet().forEach(classId -> trainingSet.put(classId, new ArrayList<>()));
//        for (int fold = 0; fold < folds.size(); fold++) {
//            if (fold == validationFold) {
//                continue;
//            }
//            folds.get(fold).forEach((k, v) -> trainingSet.get(k).addAll(v));
//        }
//        return trainingSet;
//    }
//
//    // runs each of the train/validation k-fold combination in parallel mode, requires considerably more ram than sequential mode
//    private static Summary runParallel(TextModelSupplier modelSupplier, List<Map<Double, List<String[]>>> folds) {
//        // reclaim thread pool resources since this will be very infrequent action
//        ExecutorService executorService = Executors.newWorkStealingPool();
//        List<Summary> summaries = new ArrayList<>();
//        try {
//            for (Future<Summary> future : executorService.invokeAll(prepareParallelExecutions(modelSupplier, folds))) {
//                summaries.add(future.get());
//            }
//        } catch (Exception e) {
//            throw new IllegalStateException("failed parallel execution of k-fold validation", e);
//        } finally {
//            executorService.shutdownNow();
//        }
//        return SummaryAnalysis.average(summaries);
//    }
//
//    // creates callable instances for model execution per each k fold combination
//    private static List<Callable<Summary>> prepareParallelExecutions(TextModelSupplier modelSupplier, List<Map<Double, List<String[]>>> folds) {
//        List<Callable<Summary>> modelExecutions = new ArrayList<>();
//        int validationFold = 0;
//        for (int run = 0; run < folds.size(); run++, validationFold++) {
//            modelExecutions.add(new FoldsExecutor(modelSupplier, folds, validationFold));
//        }
//        return modelExecutions;
//    }
//
//
//    /**
//     * This class is there to enable parallel execution of k-fold combinations. The lambda expressions aren't used to create
//     * {@link Callable} implementations since it's annoying to pass the different <em>validationFold</em> in a loop to each
//     * instance due to java rules for effectively final variables.
//     */
//    private static class FoldsExecutor implements Callable<Summary> {
//
//        private final TextModelSupplier modelSupplier;
//        private final List<Map<Double, List<String[]>>> folds;
//        private final int validationFold;
//
//        private FoldsExecutor(TextModelSupplier modelSupplier, List<Map<Double, List<String[]>>> folds, int validationFold) {
//            this.modelSupplier = modelSupplier;
//            this.folds = folds;
//            this.validationFold = validationFold;
//        }
//
//        @Override
//        public Summary call() {
//            Map<Double, List<String[]>> trainingSet = extractTrainingSet(folds, validationFold);
//            Map<Double, List<String[]>> validationSet = folds.get(validationFold);
//            return ModelEvaluation.execute(modelSupplier, trainingSet, validationSet);
//        }
//    }
//}
