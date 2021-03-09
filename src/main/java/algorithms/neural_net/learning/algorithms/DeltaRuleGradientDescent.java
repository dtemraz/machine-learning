package algorithms.neural_net.learning.algorithms;

import algorithms.linear_regression.optimization.real_vector.BatchGDOptimizer;
import algorithms.neural_net.learning.samples.LearningSample;
import algorithms.neural_net.learning.samples.VectorizedSamples;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements Delta rule learning backed by <strong>gradient descent</strong>. The implementation given in this
 * class alternates between batch descent and mini batch gradient descent, based on the size of learning samples.
 * The class is pretty much a proxy to {@link BatchGDOptimizer} which performs weights optimization.
 *
 * @author dtemraz
 */
public class DeltaRuleGradientDescent implements Supervisor {

    private BatchGDOptimizer batchGDOptimizer;

    public DeltaRuleGradientDescent(BatchGDOptimizer batchGDOptimizer) {
        this.batchGDOptimizer = batchGDOptimizer;
    }

    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        VectorizedSamples vectorizedSamples = VectorizedSamples.vectorize(samples);
        batchGDOptimizer.optimize(vectorizedSamples.getData(), vectorizedSamples.getOutput(), weights);
    }

}