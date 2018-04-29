package cart.optimization;

import java.util.ArrayList;

/**
 * This class implements variant of {@link SplittingOptimizer} where each feature is considered for every split.
 *
 * @author dtemraz
 */
public class FullScanOptimizer extends SplittingOptimizer {

    private final ArrayList<Integer> features = new ArrayList<>();

    /**
     * Creates instance of {@link FullScanOptimizer} which guarantees that each iteration over feature set will be
     * over entire set of features.
     *
     * @param costFunction to minimize with splits
     * @param features count of all features
     */
    public FullScanOptimizer(CostFunction costFunction, int features) {
        super(costFunction);
        // consider all features for every split
        for (int index = 0; index < features; index++) {
            this.features.add(index);
        }
    }

    @Override
    protected Iterable<Integer> getFeatures() {
        return features;
    }

}
