package algorithms.linear_regression.optimization.multiclass;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import structures.text.Vocabulary;
import utilities.math.Vector;

import java.util.List;
import java.util.Map;

/**
 * This class implements stochastic gradient descent optimization for sparse text classification with SoftMax regression.
 * The implementation assumes <strong>cross entropy cost function</strong> which makes it straightforward to define update
 * rule as ΔWk(n) = e(n) * η * x(n).
 * <p>
 * Additionally, user is able to specify L2 regularization penalty <em>lambda</em> which changes update rule into: ΔWk(n) = (e(n) * η * x(n)) - lambda*Wk
 * </p>
 *
 * @author dtemraz
 */
@RequiredArgsConstructor
@Log4j2
public class SoftMaxOptimizer implements MultiClassOptimizer {

    private final SoftMaxTextOptimizer softMaxTextOptimizer;
    private final SoftMaxVectorOptimizer softMaxVectorOptimizer;


    // by default do not log epoch errors
    public SoftMaxOptimizer(double learningRate, int epochs) {
        this(new SoftMaxTextOptimizer(learningRate, epochs, 0, true),
             new SoftMaxVectorOptimizer(learningRate, epochs, 0, true));
    }

    public SoftMaxOptimizer(double learningRate, double l2lambda, int epochs) {
        this(new SoftMaxTextOptimizer(learningRate, l2lambda, epochs),
             new SoftMaxVectorOptimizer(learningRate, epochs, 0, true));
    }

    @Override
    public void optimize(Map<Double, List<String[]>> trainingSet, Map<Double, double[]> coefficients, Vocabulary vocabulary) {
        softMaxTextOptimizer.stochastic(trainingSet, coefficients, vocabulary);
    }

    @Override
    public void optimize(Map<Double, List<double[]>> data, Map<Double, double[]> coefficients) {
        softMaxVectorOptimizer.stochastic(data, coefficients);
    }

    // prints current epoch and average epoch error across all classes
    static void printAverageEpochError(int epoch, int samples, double[][] epochError) {
        double totalError = 0;
        int classes = epochError.length;
        // calculate epoch error for each class
        for (double[] error : epochError) {
            totalError += Vector.squaredSum(error);
        }
        double averageError = totalError / classes;
        log.info("epoch: " + epoch + " , average error: " + averageError / samples);
    }

}

