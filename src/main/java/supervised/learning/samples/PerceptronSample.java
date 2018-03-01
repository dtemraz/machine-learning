package supervised.learning.samples;

import java.util.ArrayList;
import java.util.List;

/**
 * This class offers convenience methods to generate learning samples for perceptron binary classification
 * where each sample can belong to either high or low class.
 *
 * <p>The terms high and low are informal terms to define any two distinct classes for binary classification.</p>
 *
 * @author dtemraz
 */
public class PerceptronSample extends LearningSample {

    public static final double CLASS_HIGH = 1; // perceptron will output this value for all item in set 1
    public static final double CLASS_LOW = -1; // perceptron will output this value for all item in set 2

    /**
     * Creates learning samples from data split into two sets, <em>high</em> <em>low</em>.
     * All samples passed as <em>high</em> will have desired output {@link #CLASS_HIGH} and samples passed as low
     * will have desired output {@link #CLASS_LOW}.
     *
     * @param high list of samples to be considered as high class
     * @param low list of samples to be considered as low class
     * @return list of learning samples split into two sets: high and low.
     */
    public static List<LearningSample> createSamples(List<double[]> high, List<double[]> low) {
        List<LearningSample> samples = new ArrayList<>();
        high.stream().map(input -> new PerceptronSample(input, CLASS_HIGH)).forEach(samples::add);
        low.stream().map(input -> new PerceptronSample(input, CLASS_LOW)).forEach(samples::add);
        return samples;
    }

    private PerceptronSample(double[] input, double output) {
        super(input, output);
    }

}
