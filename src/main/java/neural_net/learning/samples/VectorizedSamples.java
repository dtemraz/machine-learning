package neural_net.learning.samples;

import java.util.List;

/**
 * @author dtemraz
 */
public class VectorizedSamples {

    private final double[][] data;
    private final double[] output;

    private VectorizedSamples(double[][] data, double[] output) {
        this.data = data;
        this.output = output;
    }

    public double[][] getData() {
        return data;
    }

    public double[] getOutput() {
        return output;
    }

    public static VectorizedSamples vectorize(List<LearningSample> learningSamples) {

        int samples = learningSamples.size();
        int features = learningSamples.get(0).getInput().length; // all samples will have same dimensionality

        double[][] data = new double[samples][features];
        double[] out = new double[samples];

        for (int sample = 0; sample < samples; sample++) {
            data[sample] = learningSamples.get(sample).getInput();
            out[sample] = learningSamples.get(sample).getDesiredOutput();
        }

        return new VectorizedSamples(data, out);
    }
}
