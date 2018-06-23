package ensemble;

import ensemble.model.Model;
import ensemble.model.ModelSupplier;
import ensemble.model.processor.EnsembleModelProcessor;
import ensemble.model.processor.ParallelProcessor;
import ensemble.model.processor.SequentialProcessor;
import utilities.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements stacked generalization algorithm. Set of models(level-0) are trained and their prediction result is used
 * to train {@link #combiner} model(level-1 which is used to make final prediction.
 * The assumption is that diverse enough classifiers will make uncorrelated errors which may be smoothed by another classifier
 * and improve overall performance.
 *
 * <p> The class expects that {@link #models} and {@link #combiner} implementations will be trained given a data set. </p>
 *
 * @author dtemraz
 */
public class StackedGeneralization {

    private final List<Model> models; // set of level-0 models used to make predictions
    private final Model combiner; // level-1 model used
    private final EnsembleModelProcessor ensembleModelProcessor; // models may be evaluated in parallel or sequentially
    // considers predictions, and optionally data for combiner input
    private final BiFunction<double[], double[], double[]> featureExtractor;

    /**
     * Creates instance of {@link StackedGeneralization} where <em>models</em> are used as a level-0 ensemble and
     * <em>combinerSupplier</em> as a level-1 model. Models are fitted via their constructor with <em>dataSet</em> and
     * combiner will be fitted with outputs of ensemble models.
     *
     * <p>
     *  Models will be processed in sequential mode and combiner will use only outputs of level-0 models for it's feature set.
     * </p>
     * @param modelSuppliers to train and whose predictions are used to train combiner
     * @param combinerSupplier that should instantiate combiner model given the data set
     * @param dataSet to train models
     */
    public StackedGeneralization(List<ModelSupplier> modelSuppliers, ModelSupplier combinerSupplier, List<double[]> dataSet) {
        this(modelSuppliers, combinerSupplier, dataSet, false, false);
    }

    /**
     * Creates instance of {@link StackedGeneralization} where <em>models</em> are used as a level-0 ensemble and
     * <em>combinerSupplier</em> as a level-1 model. Models are fitted via their constructor with <em>dataSet</em> and
     * combiner will be fitted with outputs of ensemble models.
     * <p>
     * The user may additionally specify if the predictions should be evaluated in parallel or sequential mode and should
     * combiner use original data set merged with ensemble prediction data set for it's training.
     * </p>
     *
     * @param modelSuppliers to train and whose predictions are used to train combiner
     * @param combinerSupplier that should instantiate combiner model given the data set
     * @param dataSet to train models
     * @param parallel defines whether predictions should be executed in parallel
     * @param includeOriginal defines whether predictions should be merged with original data set into learning set for combiner
     */
    public StackedGeneralization(List<ModelSupplier> modelSuppliers, ModelSupplier combinerSupplier, List<double[]> dataSet,
                                 boolean parallel, boolean includeOriginal) {
        this.models = new ArrayList<>();
        modelSuppliers.forEach(s -> models.add(s.get(dataSet)));
        this.ensembleModelProcessor = parallel ? ParallelProcessor::predictions : SequentialProcessor::predictions;
        this.combiner = combinerSupplier.get(buildTrainingSet(dataSet, includeOriginal));
        this.featureExtractor = includeOriginal ? Vector::merge : (data, predictions) -> predictions;
    }

    /**
     * Creates instance of {@link StackedGeneralization} where <em>models</em> are used as a level-0 ensemble and
     * <em>combinerSupplier</em> as a level-1 model. Models are fitted via their constructor with <em>sparse</em> and
     * combiner will be fitted with outputs of ensemble models and <em>dense</em> features.
     *
     * <p>The user may additionally specify if the predictions should be evaluated in parallel or sequential mode</p>
     *
     * @param modelSuppliers to train and whose predictions are used to train combiner
     * @param combinerSupplier that should instantiate combiner model given the data set
     * @param sparse features to train level-0 models
     * @param dense features to train level-1 model
     * @param parallel defines whether predictions should be executed in parallel
     */
    public StackedGeneralization(List<ModelSupplier> modelSuppliers, ModelSupplier combinerSupplier, List<double[]> sparse, List<double[]> dense, boolean parallel) {
        this.models = new ArrayList<>();
        modelSuppliers.forEach(s -> models.add(s.get(sparse)));
        this.ensembleModelProcessor = parallel ? ParallelProcessor::predictions : SequentialProcessor::predictions;
        this.combiner = combinerSupplier.get(buildTrainingSet(dense, true));
        this.featureExtractor = Vector::merge;
    }

    /**
     * Returns prediction for <em>data</em> with applied {@link #combiner} over a set of {@link #models}.
     *
     * @param data for which to make prediction
     * @return prediction which can be regression or classification for <em>data</em>
     */
    public double predict(double[] data) {
        double[] predictions = ensembleModelProcessor.predictions(models, data);
        return combiner.predict(featureExtractor.apply(data, predictions));
    }

    /**
     * Returns prediction for data which is split into <em>sparse</em> and <em>dense</em> features. Sparse features are fed
     * into level-0 models and their prediction together with dense features is used as an input to level-1 model(combiner).
     *
     * @param sparse features used as input by level-0 models
     * @param dense features used in addition with level-0 outputs as an input to level-1 model
     * @return prediction which can be regression or classification for <em>data</em>
     */
    public double predict(double[] sparse, double[] dense) {
        double[] predictions = ensembleModelProcessor.predictions(models, sparse);
        return combiner.predict(featureExtractor.apply(dense, predictions));
    }

    /**
     * Builds training set constituted of predictions made by ensemble model over <em>dataSet</em> and target classes.
     * The user may optionally specify with <em>includeOriginal</em> to aggregate built features with features in <em>dataSet</em>.
     *
     * @param dataSet from which to build predictions and training set
     * @param includeOriginal to specify if prediction features should be aggregated with <em>dataSet</em> features
     * @return training set constituted of predictions made by ensemble model over <em>dataSet</em> and target classes,
     * optionally merged with <em>dataSet</em>
     */
    private List<double[]> buildTrainingSet(List<double[]> dataSet, boolean includeOriginal) {
        List<double[]> predictionSet = new ArrayList<>(); // ensemble predictions and target class
        for (double[] row : dataSet) {
            // prediction of each model for a given data row
            double[] predictions = ensembleModelProcessor.predictions(models, row);
            // this array should contains predictions of each model and a target class
            double[] withExpectedClass = new double[predictions.length + 1];
            System.arraycopy(predictions, 0, withExpectedClass, 0, predictions.length);
            // last item in row contains target class
            withExpectedClass[predictions.length] = row[models.size() - 1];
            predictionSet.add(withExpectedClass);
        }

        // user may opt to merge prediction features with existing features
        if (!includeOriginal) {
            return predictionSet;
        }

        int features = dataSet.get(0).length - 1; // number of features in original set without class
        int modeled = predictionSet.get(0).length; // features in prediction set with class
        List<double[]> aggregateSet = new ArrayList<>();
        for (int row = 0; row < dataSet.size(); row++) {
            double[] aggregated = new double[features + modeled];
            // copy original features
            System.arraycopy(dataSet.get(row), 0, aggregated, 0, features);
            // aggregate prediction features and target class with original features
            System.arraycopy(predictionSet.get(row), 0, aggregated, features, modeled);
            aggregateSet.add(aggregated);
        }

        return aggregateSet;
    }

}
