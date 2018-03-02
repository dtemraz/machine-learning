package optimization;

import java.util.Arrays;

/**
 * This class defines early exit criteria which should cause learning procedure to stop if the criteria is true.
 * This is generally useful strategy to combat overfitting. The strategies in this interface expect only epoch error of a
 * training set.
 * <p>
 * Ideally, there should also be method which accepts epoch error of a validation set so that we can terminate
 * learning in cross validation mode(which is not supported anyway).
 * </p>.
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface StoppingCriteria {

    /**
     * Returns true if the learning should stop given this <em>epochError</em>, false otherwise.
     *
     * @param epochError to check and determine if learning should stop
     * @return true if the learning should stop given this <em>epochError</em>, false otherwise
     */
    boolean test(double[] epochError);

    /**
     * Returns true if all errors in this <em>epochError</em> are bellow <em>errorTolerance</em>, false otherwise.
     *
     * @param errorTolerance all errors in epoch should be less than errorTolerance
     * @return true if all errors in this <em>epochError</em> are bellow <em>errorTolerance</em>, false otherwise
     */
    static StoppingCriteria allBellowTolerance(double errorTolerance) {
        return epochError -> !Arrays.stream(epochError).filter(e -> Math.abs(e) > errorTolerance).findFirst().isPresent();
    }

    /**
     * Returns true if squared error sum of all errors in <em>epochError</em> is bellow <em>errorTolerance</em>, false otherwise.
     *
     * @param errorTolerance squared error sum of all errors in epoch should be less than <em>errorTolerance</em>
     * @return true if squared error sum of all errors in <em>epochError</em> is bellow <em>errorTolerance</em>, false otherwise.
     */
    static StoppingCriteria squaredErrorBellowTolerance(double errorTolerance) {
        return epochError -> Arrays.stream(epochError).map(e -> e * e).reduce(Double::sum).getAsDouble() < errorTolerance;
    }

}