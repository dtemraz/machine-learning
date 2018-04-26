package ensemble;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * This class lets user create random sample uniformly and with replacement for a given data. The with replacement simply
 * means that a given row can be extracted multiple times from data set.
 *
 * @author dtemraz
 */
public class Sampler {

    private final Random random = new Random();

    /**
     * Creates a uniform and with replacement sample from a <em>dataSet</em> with size equal to <em>ratio</em> of the
     * data set size.
     *
     * @param dataSet to sample
     * @param ratio percentage of <em>dataSet</em> size to use as a sample size
     * @return uniform with replacement sample from <em>dataSet</em> with size equal to <em>ratio</em> of data set size
     */
    public List<double[]> subset(List<double[]> dataSet, double ratio) {
        List<double[]> subset = new ArrayList<>();
        int subsetSize = (int) (dataSet.size() * ratio);
        for (int row = 0; row < subsetSize; row++) {
            // allow a sample to be used multiple times in a subset
            subset.add(dataSet.get(random.nextInt(dataSet.size())));
        }
        return subset;
    }

}
