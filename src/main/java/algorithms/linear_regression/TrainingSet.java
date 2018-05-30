package algorithms.linear_regression;

import java.util.Arrays;
import java.util.List;

/**
 * This class is a simple utility class which maps {@link  List} of doubles into 2-d array.
 * For a user, it should be more convenient to pass the samples with a list than to pass them with 2-d array.
 * Arrays are more convenient(and performs better) for internal use in {@link optimization.GradientDescent}.
 *
 * @author dtemraz
 */
class TrainingSet {

    final double[][] input; // array of input samples for training
    final double[] expected; // array of associated expected class for input samples, with matching indexes

    TrainingSet(double[][] input, double[] expected) {
        this.input = input;
        this.expected = expected;
    }

    static TrainingSet build(List<double[]> trainingSet) {
        int features = trainingSet.get(0).length;
        int samples = trainingSet.size();
        int classIndex = features - 1;

        double[][] trainingSamples = new double[samples][features];
        double[] values = new double[samples];

        for (int sample = 0; sample < samples; sample++) {
            double[] trainingData = trainingSet.get(sample);
            // copy all elements except class id into input field
            trainingSamples[sample] = Arrays.copyOf(trainingData, classIndex);
            // copy expected class id for the sample into expected field
            values[sample] = trainingData[classIndex];
        }
        return new TrainingSet(trainingSamples, values);
    }

}
