package ensemble;

import ensemble.model.Model;
import ensemble.model.ModelSupplier;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

/**
 * This class defines bootstrap aggregation, also known as bagging, which is an algorithm to reduce variance error. It can be
 * used with any {@link Model} however it is expected to give best results on models with low bias and <strong>high</strong>
 * variance, such as decision trees.
 *
 *<p>
 * Machine learning algorithms attempt to find the right balance of bias error and variance error. Bias error is introduced
 * with our assumptions(restrictions) about target functions, while variance error is due to model's susceptibility to change
 * target function when there are even small changes in data set.*
 * Consider simple linear regression: Y=b0+b1. This model places tight restrictions on the form of a target function, therefore
 * it's a <strong>high bias model</strong>. On the other hand, due to <strong>low variance</strong> the target function is not that much sensitive to small
 * changes in training set.
 * </p>
 *
 * The class offers method {@link #bootstrap(ModelSupplier, List)} which bootstraps data set and fits models with
 * bootstrapped samples, the method {@link #classify(double[])} to make classifications by majority of votes among fitted models
 * and {@link #estimate(double[])} to perform regression as average of fitted models regression.
 *
 * @author dtemraz
 */
public class BootstrapAggregation {

    private final double resampleRatio; // percentage of original sample that should be used as sub sample size
    private final int subsets; // number of subsets(samples) to derive from original set
    private final List<Model> ensembleModel; // list of classifiers whose prediction will be aggregated in final result
    private final Sampler sampler; // utility class that creates sample from a given data set

    public BootstrapAggregation(double resampleRatio, int subsets) {
        this.resampleRatio = resampleRatio;
        this.subsets = subsets;
        ensembleModel = new ArrayList<>();
        sampler = new Sampler();
    }

    /**
     * Bootstraps <em>dataSet</em> with {@link #subsets} and creates a model for which of the subsets. Such ensemble model
     * can be used for prediction by majority of votes or regression by averaging the estimation.
     *
     * @param supplier which creates instance(s) of a model
     * @param dataSet which will be sampled and fed into models
     */
    public void bootstrap(ModelSupplier supplier, List<double[]> dataSet) {
        bootstrap(dataSet).forEach(sample -> ensembleModel.add(supplier.get(sample)));
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
     * returns regression of target value for <em>data</em> by averaging of ensemble model outputs.
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
        // let each model vote for a class and count votes per class
        HashMap<Double, Integer> votingResults = new HashMap<>();
        ensembleModel.stream().forEach(model -> votingResults.merge(model.predict(data), 1, (old, n) -> old + n));

        // find class for which majority of models voted
        return votingResults.entrySet().stream()
                .max(Comparator.comparingInt(c -> c.getValue()))
                .get()
                .getKey();
    }

    // bootstraps this data set into subsets as defined in constructor parameters
    private List<List<double[]>> bootstrap(List<double[]> dataSet) {
        List<List<double[]>> samples = new ArrayList<>();
        for (int subset = 0; subset < subsets; subset++) {
            samples.add(sampler.subset(dataSet, resampleRatio));
        }
        return samples;
    }

}