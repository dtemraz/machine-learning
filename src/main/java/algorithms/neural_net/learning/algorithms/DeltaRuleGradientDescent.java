package algorithms.neural_net.learning.algorithms;

import algorithms.neural_net.learning.samples.LearningSample;
import algorithms.neural_net.learning.samples.VectorizedSamples;
import algorithms.linear_regression.optimization.real_vector.GradientDescent;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements Delta rule learning backed by <strong>gradient descent</strong>. The implementation given in this
 * class alternates between batch descent and mini batch gradient descent, based on the size of learning samples.
 * The class is pretty much a proxy to {@link GradientDescent} which performs weights optimization.
 *
 * @author dtemraz
 */
public class DeltaRuleGradientDescent implements Supervisor {

    private final GradientDescent gradientDescent;
    private static final int MINI_BATCH_CUTOFF = 50; // if there are more samples than this, do mini batch gradient descent.
    private static final double SIZE_SPLIT_FACTOR = 0.25;

    public DeltaRuleGradientDescent(GradientDescent gradientDescent) {
        this.gradientDescent = gradientDescent;
    }

    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        VectorizedSamples vectorizedSamples = VectorizedSamples.vectorize(samples);
        if (samples.size() < MINI_BATCH_CUTOFF) {
            gradientDescent.batch(vectorizedSamples.getData(), vectorizedSamples.getOutput(), weights, neuronOutput::apply);
        } else {
            gradientDescent.miniBatch(vectorizedSamples.getData(), vectorizedSamples.getOutput(), weights, ((int)(samples.size() * SIZE_SPLIT_FACTOR)), neuronOutput::apply);
        }
    }

}