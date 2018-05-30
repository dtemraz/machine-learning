package algorithms.cart.optimization;
import structures.RandomizedQueue;

import java.util.Iterator;

/**
 * This class implements variant of {@link SplittingOptimizer} where for every split, <strong>subset</strong> of features
 * is considered.
 *
 * @author dtemraz
 */
public class RandomFeaturesOptimizer extends SplittingOptimizer {

    private final int candidates;

    private final RandomizedQueue<Integer> randomizedQueue = new RandomizedQueue<>();

    /**
     * Creates instance of {@link RandomFeaturesOptimizer} which guarantees that each iteration over feature set will be
     * over random subset of features with size <em>candidates</em>.
     *
     * @param costFunction to minimize with splits
     * @param features count of all features
     * @param candidates number of features to consider in subset
     */
    public RandomFeaturesOptimizer(CostFunction costFunction, int features, int candidates) {
        super(costFunction);
        this.candidates = candidates;
        // add all feature indexes, but each iterator instance will only iterate over subset
        for (int x = 0; x < features; x++) {
            randomizedQueue.enqueue(x);
        }
    }

    // iterator which guarantees non deterministic order on every iteration
    private Iterable<Integer> randomFeatures() {
        return () -> new Iterator<Integer>() {
            private int cnt = 0;
            private final Iterator<Integer> it = randomizedQueue.iterator();

            @Override
            public boolean hasNext() {
                return cnt < candidates && it.hasNext();
            }

            @Override
            public Integer next() {
                cnt++;
                return it.next();
            }
        };
    }

    @Override
    protected Iterable<Integer> getFeatures() {
        return randomFeatures();
    }
}