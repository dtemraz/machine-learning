package ensemble;

import ensemble.model.Model;
import ensemble.model.ModelSupplier;
import utilities.Sampler;

import java.util.ArrayList;
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
 * it's a <strong>high bias model</strong>. On the other hand, due to <strong>low variance</strong> the target function is
 * not that much sensitive to small changes in training set.
 * </p>
 *
 * The class offers shall bootstraps data set and fits models with bootstrapped samples via constructor,
 * the method {@link #classify(double[])} to make classifications by majority of votes among fitted models and
 * {@link #estimate(double[])} to perform regression as average of fitted models regression.
 *
 * @author dtemraz
 */
public class BootstrapAggregation {

    private final double resampleRatio; // percentage of original sample that should be used as sub sample size
    private final int subsets; // number of subsets(samples) to derive from original set
    private final CommitteeOfExperts committeeOfExperts;

    public BootstrapAggregation(ModelSupplier supplier, List<double[]> dataSet, double resampleRatio, int subsets) {
        this.resampleRatio = resampleRatio;
        this.subsets = subsets;
        List<Model> ensembleModel = new ArrayList<>();
        bootstrap(dataSet).forEach(sample -> ensembleModel.add(supplier.get(sample)));
        committeeOfExperts = new CommitteeOfExperts(ensembleModel);
    }

    /**
     * Returns expected class for <em>data</em> by majority of votes in ensemble model.
     *
     * @param data to classify
     * @return expected class for <em>data</em> by majority of votes in ensemble model
     */
    public double classify(double[] data) {
        return committeeOfExperts.classify(data);
    }

    /**
     * Returns regression of target value for <em>data</em> by averaging of ensemble model outputs.
     *
     * @param data for which to estimate target value
     * @return regression of target value for <em>data</em> by averaging of ensemble model outputs
     */
    public double estimate(double[] data) {
        return committeeOfExperts.estimate(data);
    }

    // bootstraps this data set into subsets as defined in constructor parameters
    private List<List<double[]>> bootstrap(List<double[]> dataSet) {
        return new Sampler().bootstrap(dataSet, resampleRatio, subsets);
    }

}