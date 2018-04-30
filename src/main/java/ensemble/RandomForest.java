package ensemble;

import cart.ClassificationTree;
import cart.optimization.CostFunction;
import cart.optimization.RandomFeaturesOptimizer;
import ensemble.model.Model;
import utilities.Sampler;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * This class implements Random Forest algorithm, an extension of {@link BootstrapAggregation}. The algorithm attempts to reduce
 * high collinear between bootstrapped classification trees by introducing variability in feature selection.
 * <p>
 * Each tree in forest
 * should only operate on subset of features which should make them more diverse and robust as a whole.
 * Therefore, the only change this algorithm introduces to bootstrapping aggregation is in the number of features considered
 * for splitting, with {@link RandomFeaturesOptimizer}.
 * </p>
 *
 * Once trained, trees make classification via majority voting procedure through {@link CommitteeOfExperts}.
 *
 * <p>
 * <strong>Note:</strong> While this implementation works in regression setting, i have yet to add regression tree implementation.
 * </p>
 *
 * @author dtemraz
 */
public class RandomForest {

    // this is highest recommend setting in the algorithm specification
    private static final Function<Integer, Integer> DEFAULT_CANDIDATES = features -> 2 * (int) Math.sqrt(features);
    private static final int DEPTH = 10; // default maximal depth allowed for any given tree
    private static final int MIN_SPLIT_NODES = 10; // default minimal number of nodes to be considered for a split

    private final double resampleRatio; // percentage of original sample that should be used as sub sample size
    private final int trees; // number of subsets(samples) to derive from original set
    private final CommitteeOfExperts committeeOfExperts; // ensemble model in which individual trees vote for prediction

    public RandomForest(List<double[]> dataSet, double resampleRatio, int trees) {
        this(dataSet, resampleRatio, trees, DEFAULT_CANDIDATES);
    }

    public RandomForest(List<double[]> dataSet, double resampleRatio, int trees, Function<Integer, Integer> candidates) {
        this.resampleRatio = resampleRatio;
        this.trees = trees;
        int features = dataSet.get(0).length - 1;
        RandomFeaturesOptimizer optimizer = optimizer(features, candidates.apply(features));
        List<Model> ensembleModel = new ArrayList<>();
        bootstrap(dataSet).forEach(sample ->
                ensembleModel.add(new ClassificationTree(sample, optimizer, MIN_SPLIT_NODES, DEPTH)::classify));
        committeeOfExperts = new CommitteeOfExperts(ensembleModel, false);
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
        return new Sampler().bootstrap(dataSet, resampleRatio, trees);
    }

    private static RandomFeaturesOptimizer optimizer(int features, int candidates) {
        return new RandomFeaturesOptimizer(CostFunction.GINI_INDEX, features, candidates);
    }

}
