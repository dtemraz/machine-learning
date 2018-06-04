package algorithms.linear_regression.optimization.text;

/**
 * This class defines early exit criteria which should stop learning procedure given sum of squared epoch error components
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface SquaredErrorStoppingCriteria {

    /**
     * Returns true if the learning should stop given this <em>squaredEpochError</em>, false otherwise.
     *
     * @param squaredEpochError sum of squared epoch error components to check and determine if learning should stop
     * @return true if the learning should stop given this <em>squaredEpochError</em>, false otherwise
     */
    boolean test(double squaredEpochError);

    /**
     * Returns true if squared sum of epoch error components is bellow <em>errorTolerance</em>, false otherwise.
     *
     * @param errorTolerance squared sum of epoch error components should be less than <em>errorTolerance</em>
     * @return true if squared sum of epoch error components is bellow <em>errorTolerance</em>, false otherwise
     */
    static SquaredErrorStoppingCriteria squaredErrorBellowTolerance(double errorTolerance) {
        return squaredEpochError -> squaredEpochError < errorTolerance;
    }

}