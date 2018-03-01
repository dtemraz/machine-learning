package examples.and;

import supervised.learning.samples.LearningSample;
import supervised.learning.samples.PerceptronSample;
import supervised.learning.algorithms.DeltaRuleGradientDescent;
import supervised.learning.algorithms.DeltaRuleStochasticGradientDescent;
import supervised.learning.algorithms.Perceptron;
import supervised.neuron.Activation;
import supervised.neuron.Neuron;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class TestAnd {

    private static final int AND_FUNC_VARS = 2;

    private static final Function<Double, Double> _1_PER_CENT_ERROR_CLASSIFICATION = x -> (x > 0.99) ? 1D : x < 0.01 ? 0 : -1;

    public static void main(String[] args) {
        
        System.out.println("## running tests with perceptron ##");
        Neuron perceptron = new Neuron(AND_FUNC_VARS, Activation.SIGNUM::apply, new Perceptron());
        run(perceptron, PerceptronSample.CLASS_LOW, PerceptronSample.CLASS_HIGH);

        System.out.println("## running tests with delta rule gradient descent ##");
        Neuron deltaGD = new Neuron(AND_FUNC_VARS, Activation.SIGMOID::apply, new DeltaRuleGradientDescent(0.3, 0.00009, 100_000), _1_PER_CENT_ERROR_CLASSIFICATION);
        run(deltaGD, 0, 1);

        System.out.println("## running tests with delta rule stochastic gradient descent ##");
        Neuron deltaSGD = new Neuron(AND_FUNC_VARS, Activation.SIGMOID::apply, new DeltaRuleStochasticGradientDescent(0.2, 0.009, 100_000), _1_PER_CENT_ERROR_CLASSIFICATION);;
        run(deltaSGD, 0, 1);
    }

    private static void run(Neuron neuron, double classOne, double classTwo) {
        List<LearningSample> learningSamples = getLearningSamples(classOne, classTwo);
        Collections.shuffle(learningSamples);

        neuron.train(learningSamples);
        
        Collections.shuffle(learningSamples);
        learningSamples.forEach(sample -> verifyClass(sample, neuron));
    }

    private static void verifyClass(LearningSample sample, Neuron neuron) {
        System.out.println("desired: " + sample.getDesiredOutput() + " , got: " + neuron.output(sample));
        if (sample.getDesiredOutput() != neuron.output(sample)) {
            throw new IllegalStateException("wrong classification: " + neuron.output(sample));
        }
    }

    // (-1, 1) for perceptron,  (0, 1) for delta rule
    private static List<LearningSample> getLearningSamples(double classOne, double classTwo) {
        return Arrays.asList(new LearningSample(new double[] { 0, 0 }, classOne),
                             new LearningSample(new double[] { 0, 1 }, classOne),
                             new LearningSample(new double[] { 1, 0 }, classOne),
                             new LearningSample(new double[] { 1, 1 }, classTwo));
    }
    
}
