package examples.iris;

import algorithms.k_means.K_Means;
import algorithms.k_means.Member;
import algorithms.linear_regression.optimization.real_vector.BatchGDOptimizer;
import algorithms.linear_regression.optimization.real_vector.StoppingCriteria;
import algorithms.neural_net.Activation;
import algorithms.neural_net.Neuron;
import algorithms.neural_net.learning.algorithms.DeltaRuleGradientDescent;
import algorithms.neural_net.learning.algorithms.DeltaRuleStochasticGradientDescent;
import algorithms.neural_net.learning.algorithms.Perceptron;
import algorithms.neural_net.learning.samples.LearningSample;
import algorithms.neural_net.learning.samples.PerceptronSample;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TestIris {

    private static final File IRIS_LEARNING_SAMPLES = new File("src/test/resources/Iris_learning");
    private static final File IRIS_VALIDATION_SAMPLES = new File("src/test/resources/Iris_validation");

    private static final Function<Double, Double> _1_PER_CENT_ERROR_CLASSIFICATION = x -> (x > 0.99) ? 1D : x < 0.01 ? 0 : -1;

    public static void main(String[] args) {

        System.out.println("## running tests with perceptron ##");
        Neuron perceptron = new Neuron(IrisReader.IRIS_MEASURES, Activation.SIGNUM::apply, new Perceptron());

        run(perceptron, PerceptronSample.CLASS_HIGH, PerceptronSample.CLASS_LOW);

        System.out.println();

        System.out.println("## running tests with delta rule stochastic gradient descent ##");
        Neuron deltaSGD = new Neuron(IrisReader.IRIS_MEASURES, Activation.SIGMOID::apply, new DeltaRuleStochasticGradientDescent(0.2, 0.0009, 100_000), _1_PER_CENT_ERROR_CLASSIFICATION);
        run(deltaSGD, 0, 1);

        System.out.println();

        System.out.println("## running tests with delta rule gradient descent ##");
        BatchGDOptimizer gd = new BatchGDOptimizer(0.2, 100_000, 10, StoppingCriteria.squaredErrorBellowTolerance(0.000009));

        Neuron deltaGD = new Neuron(IrisReader.IRIS_MEASURES, Activation.SIGMOID::apply, new DeltaRuleGradientDescent(gd), _1_PER_CENT_ERROR_CLASSIFICATION);
        run(deltaGD, 0, 1);

        System.out.println();

        System.out.println("## running tests with k-means ##");
        testKMeans(perceptron);

    }

    private static void run(Neuron neuron, double setosaClass, double versicolorClass) {
        // prepare learning data
        List<LearningSample> learningSamples = extractSamples(IrisReader.read(IRIS_LEARNING_SAMPLES), setosaClass, versicolorClass);
        Collections.shuffle(learningSamples);

        // train neuron
        neuron.train(learningSamples);

        // verify known learning samples
        System.out.println("verifying seen samples");
        Collections.shuffle(learningSamples);
        verifySamples(learningSamples, neuron);

        System.out.println();

        // verify unseen samples
        System.out.println("verifying unseen  samples");
        List<LearningSample> validationSamples = extractSamples(IrisReader.read(IRIS_VALIDATION_SAMPLES), setosaClass, versicolorClass);
        Collections.shuffle(validationSamples);

        verifySamples(validationSamples, neuron);
    }

    private static void verifySamples(List<LearningSample> samples, Neuron neuron) {
        long successful = samples.stream().filter(sample -> neuron.output(sample) == sample.getDesiredOutput()).count();
        System.out.println("correctly classified: " + successful);
        System.out.println("incorrectly classified: " + (samples.size() - successful));
    }


    // ---------------------------------------------------

    private static List<LearningSample> extractSamples(Map<IrisType, List<Iris>> irisTypes, double setosaClass, double versicolorClass) {
        List<LearningSample> samples = mapToSamples(irisTypes.get(IrisType.SETOSA), setosaClass);
        samples.addAll(mapToSamples(irisTypes.get(IrisType.VERSICOLOR), versicolorClass));
        return samples;
    }

    // maps iris collection to learning samples with desired output equal to classValue
    private static List<LearningSample> mapToSamples(List<Iris> irisCollection, double classValue) {
        return irisCollection.stream()
                             .map(iris -> new LearningSample(iris.getComponents(), classValue))
                             .collect(Collectors.toList());
    }

    private static Double[] boxInput(LearningSample sample) {
        Double[] wrapper = new Double[4];
        double[] input = sample.getInput();
        for (int i = 1; i <= 4; i++) {
            wrapper[i - 1] = input[i];
        }
        return wrapper;
    }

    private static double[] unboxInput(Double[] wrapper) {
        double[] input = new double[4];
        for (int i = 0; i < 4; i++) {
            input[i] = wrapper[i];
        }
        return input;
    }


    private static void testKMeans(Neuron neuron) {
        List<LearningSample> learningSamples = readAllSamples();
        List<Double[]> centroids = initializeCentroids(learningSamples);

        Collections.shuffle(learningSamples);
        List<Double[]> clusterSamples = learningSamples.stream().map(TestIris::boxInput).collect(Collectors.toList());

        K_Means k_means = new K_Means(centroids);
        k_means.cluster(clusterSamples);

        verifyCluster(k_means, neuron, 0);
        verifyCluster(k_means, neuron, 1);

    }

    private static List<LearningSample> readAllSamples() {
        List<LearningSample> learningSamples = extractSamples(IrisReader.read(IRIS_LEARNING_SAMPLES), 0, 1);
        learningSamples.addAll(extractSamples(IrisReader.read(IRIS_VALIDATION_SAMPLES), 0, 1));
        return learningSamples;
    }

    private static List<Double[]> initializeCentroids(List<LearningSample> learningSamples) {
        int last = learningSamples.size() - 1;
        List<Double[]> centroids = new ArrayList<>();
        centroids.add(boxInput(learningSamples.get(0)));
        centroids.add(boxInput(learningSamples.get(last)));
        learningSamples.remove(0);
        learningSamples.remove(learningSamples.size() - 1);
        return centroids;
    }

    private static void verifyCluster(K_Means k_means, Neuron neuron, int clusterId) {
        int setosaCount = 0;
        int versicolorCount = 0;
        for (Member member : k_means.membersFor(clusterId)) {
            Double[] data = member.getData();
            if (neuron.output(unboxInput(data)) == 1) {
                setosaCount++;
            } else { versicolorCount++; }
        }
        System.out.println("cluster id: " + clusterId + " , setosa count: " + setosaCount + ", versicolor count: " + versicolorCount);
    }

}