package ensemble;

import ensemble.model.Model;
import ensemble.model.processor.EnsembleModelProcessor;
import ensemble.model.processor.ParallelProcessor;
import ensemble.model.processor.SequentialProcessor;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class implements majority based voting classification and regression. This is the most democratic approach to voting
 * where each classifier, be it weak or strong, has same influence on final decision.
 * <p>
 * The class supports parallel execution for classification tasks via constructor configuration:
 * {@link #CommitteeOfExperts(List, boolean)} where boolean value defines should classification be performed in parallel
 * or sequential.
 * </p>
 *
 * The class offers two methods, {@link #classify(double[])} which returns most voted class given ensemble model of
 * classifiers and {@link #estimate(double[])} which estimates explanatory variables as average of ensemble model classifiers.
 *
 * @author dtemraz
 */
public class CommitteeOfExperts {

    private final EnsembleModelProcessor ensembleModelProcessor; // sequential or parallel model execution is supported
    private final List<Model> ensembleModel;  // ensemble classifier where each classifier is equally important

    public CommitteeOfExperts(List<Model> ensembleModel) {
        this(ensembleModel, false);
    }

    public CommitteeOfExperts(List<Model> ensembleModel, boolean parallelExecution) {
        this.ensembleModel = ensembleModel;
        this.ensembleModelProcessor = parallelExecution ? ParallelProcessor::predictions : SequentialProcessor::predictions;
    }

    /**
     * Returns expected class for <em>data</em> by majority of votes in ensemble model.
     *
     * @param data to classify
     * @return expected class for <em>data</em> by majority of votes in ensemble model
     */
    public double classify(double[] data) {
        return vote(data);
    }

    /**
     * Returns regression of target value for <em>data</em> by averaging of ensemble model outputs.
     *
     * @param data for which to estimate target value
     * @return regression of target value for <em>data</em> by averaging of ensemble model outputs
     */
    public double estimate(double[] data) {
        return ensembleModel.stream().mapToDouble(m -> m.predict(data))
                .reduce(Double::sum)
                .getAsDouble() / ensembleModel.size();
    }

    // each model in ensemble votes for data class and the class with most votes is chosen as target class
    private double vote(double[] data) {
        double[] predictions = ensembleModelProcessor.predictions(ensembleModel, data);
        HashMap<Double, Long> votes = new HashMap<>();
        for (double prediction : predictions) {
            votes.merge(prediction, 1L, (old, val) -> old + val);
        }
        return  mostOccurrences(votes);
    }

    /**
     * The class implements parallel ensemble model evaluation with the work stealing semantics. Maybe a bit unintuitive,
     * but the class doesn't use merge step. Instead result of each model is stored in concurrent queue which is evaluated
     * when all tasks are executed.
     * Each element of the queue contains prediction from a given model.
     */

    // returns whichever key has highest frequency(value) count
    private double mostOccurrences(HashMap<Double, Long> votingResults) {
        return votingResults.entrySet().stream()
                .max(Comparator.comparingLong(Map.Entry::getValue))
                .get()
                .getKey();
    }

}