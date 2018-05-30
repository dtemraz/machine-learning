package algorithms.neural_net.learning.algorithms;

import algorithms.neural_net.learning.samples.LearningSample;
import utilities.Vector;

import java.util.List;
import java.util.function.BiFunction;

/**
 * This class implements Perceptron learning rule that teaches neuron to perform binary classification.
 * The algorithm is guaranteed to converge for linearly separable problems.
 * For non-linearly separable problems, this learning procedure will never stop.
 *
 * <p>
 * The perceptron finds n-1 dimensional hyperplane that splits n-dimensional data set X in two subsets: X1, X2 such that
 * all xϵX1 are on one side of plane and all xϵX2 on the other.
 * </p>
 *
 * Algorithm steps:
 * <ul>
 *   <li> 1.  present input and calculate output class </li>
 *   <li> 2.  if correctly classified, go to next sample </li>
 *   <li> 2a. if activation to high, reduce weights(d = - 1) according to weight adjustment rule Δ Wk(n)</li>
 *   <li> 2b. if activation to small, increase weights(d = 1) according to weight adjustment rule Δ Wk(n)</li>
 *   <li> 3.  repeat until samples are correctly classified </li>
 * </ul>
 *
 * where weight adjustment rule => Δ Wk(n) = d * η * x(n)
 *
 * <ul>
 *  <li> d = correct class for a given input sample:  +1 | -1 </li>
 *  <li> η = {@link #learningRate}</li>
 *  <li> x(n) = input vector instance </li>
 * </ul>
 *
 * @author dtemraz
 *
 */
public class Perceptron implements Supervisor {

    private static final double DEFAULT_LEARNING_RATE = 0.20;

    private final double learningRate;

    public Perceptron() {
        this(DEFAULT_LEARNING_RATE);
    }

    public Perceptron(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void train(List<LearningSample> samples, double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        boolean converged = false; // converges when all samples are correctly classified
        int epoch = 0;
        while (!converged) {
            epoch++;
            converged = true;
            for (LearningSample sample : samples) {
                if (!correct(sample, weights, neuronOutput)) {
                    converged = false; // there is at least one wrongly classified sample in this epoch
                    adjustWeights(sample, weights);
                }
            }
        }
        System.out.println(String.format("converged in %d epoch", epoch));
    }

    // returns true if sample was correctly classified, false otherwise
    private boolean correct(LearningSample sample,  double[] weights, BiFunction<double[], double[], Double> neuronOutput) {
        return neuronOutput.apply(sample.getInput(), weights) == sample.getDesiredOutput();
    }

    // modifies weights according to Perceptron learning rule: Δ Wk(n) = d * η * x(n)
    // see DeltaRuleStochasticGradientDescent for version with loop
    private void adjustWeights(LearningSample sample, double[] weights) {
        // this is the d * η from the above equation, it is constant for all weights
        double delta = sample.getDesiredOutput() * learningRate;
        // each weight should be modified proportionally to input signal it receives from the sample, this is the * x(n) term
        double[] gradient = Vector.multiply(sample.getInput(), delta);
        // adds gradient to previous value of weights vector and that is how we get updated weight
        Vector.mergeSum(weights, gradient);
    }

}
