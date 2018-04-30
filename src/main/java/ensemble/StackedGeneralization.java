package ensemble;

import ensemble.model.processor.EnsembleModelProcessor;
import ensemble.model.Model;
import ensemble.model.ModelSupplier;
import ensemble.model.processor.ParallelProcessor;
import ensemble.model.processor.SequentialProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * @author dtemraz
 */
public class StackedGeneralization {

    private final List<Model> models;
    private final Model combiner;
    private final EnsembleModelProcessor ensembleModelProcessor;

    public StackedGeneralization(List<Model> models, ModelSupplier combinerSupplier, List<double[]> dataSet) {
        this(models, combinerSupplier, dataSet, false);
    }

    public StackedGeneralization(List<Model> models, ModelSupplier combinerSupplier, List<double[]> dataSet, boolean parallel) {
        this.models = models;
        this.ensembleModelProcessor = parallel ? ParallelProcessor::predictions : SequentialProcessor::predictions;
        this.combiner = combinerSupplier.get(buildTrainingSet(dataSet));
    }

    /**
     * Returns expected class for <em>data</em> by majority of votes in ensemble model.
     *
     * @param data to classify
     * @return expected class for <em>data</em> by majority of votes in ensemble model
     */
    public double predict(double[] data) {
        return combiner.predict(ensembleModelProcessor.predictions(models, data));
    }

    private List<double[]> buildTrainingSet(List<double[]> dataSet) {
        List<double[]> trainingSet = new ArrayList<>();
        for (double[] row : dataSet) {
            // prediction of each model for a given data row
            double[] predictions = ensembleModelProcessor.predictions(models, row);
            // this array should contains predictions of each model and a target class
            double[] withExpectedClass = new double[predictions.length + 1];
            System.arraycopy(predictions, 0, withExpectedClass, 0, predictions.length);
            // last item in row contains target class
            withExpectedClass[predictions.length] = row[models.size() - 1];
            trainingSet.add(withExpectedClass);
        }
        return trainingSet;
    }

}
