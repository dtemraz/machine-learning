package algorithms.neural_net;

import java.util.function.Function;

/**
 * This enum offers implementation of some of the common activation functions(and their derivatives) 
 * used in neural networks.
 * Each enum class offers method {@link #apply(double)} that calculates value for a function in some point
 * and {@link #derivative()} that returns derivative for the function.
 * 
 * @author dtemraz
 */
public enum Activation {

    IDENTITY {
        @Override
        public double apply(double x) {
            return x;
        }
        @Override
        public Function<Double, Double> derivative() {
            return x -> 1D;
        }
    },

    RELU {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }
        @Override
        public Function<Double, Double> derivative() {
            return x -> (x > 0) ? x : 0;
        }
    },
    
    SIGMOID {
        @Override
        public double apply(double x) {
            return 1 / (1 + Math.pow(Math.E, -x));
        }
        @Override
        public Function<Double, Double> derivative() {
            return x -> apply(x) * (1 - apply(x));
        }
    },
    
    SIGNUM {
        @Override
        public double apply(double x) {
            return x > 0 ? 1 : -1;
        }
        @Override
        public Function<Double, Double> derivative() {
            throw new UnsupportedOperationException();
        }
    };
    
    /**
     * Calculates function value for x.
     * 
     * @param x for which to calculate function value
     * @return function value in x
     */
    public abstract double apply(double x);
    
    /**
     * Returns derivative of activation function.
     * 
     * @return derivative of activation function
     */
    public abstract Function<Double, Double> derivative();
}
