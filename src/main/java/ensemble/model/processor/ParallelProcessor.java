package ensemble.model.processor;

import ensemble.model.Model;

import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * @author dtemraz
 */
public class ParallelProcessor {

    private static final double EXECUTION_CUT_OFF = 0.05;

    private static final ForkJoinPool executor = new ForkJoinPool(); // could have been in lambda body, but lazy initialization is slow

    private ParallelProcessor() {

    }

    public static double[] predictions(List<Model> ensemble, double[] data) {
        ConcurrentLinkedQueue<Double> predictions = new ConcurrentLinkedQueue<>();
        executor.submit(new ModelEvaluation(0, ensemble.size() - 1, data, predictions, ensemble)).join();
        double[] results = new double[ensemble.size()];
        for (int prediction = 0; prediction < results.length; prediction++) {
            results[prediction] = predictions.poll();
        }
        return results;
    }

    /**
     * The class implements parallel ensemble model evaluation with the work stealing semantics. Maybe a bit unintuitive,
     * but the class doesn't use merge step. Instead result of each model is stored in concurrent queue which is evaluated
     * when all tasks are executed.
     * Each element of the queue contains prediction from a given model.
     */
    private static class ModelEvaluation extends RecursiveAction {

        private final int from; // first model that should be evaluated by this task
        private final int to; // last model that should be evaluated by this task
        private final double[] data; // data for which to make prediction
        private final ConcurrentLinkedQueue<Double> queue; // stores prediction of each model

        private final List<Model> ensemble;

        private ModelEvaluation(int from, int to, double[] data, ConcurrentLinkedQueue<Double> queue, List<Model> ensemble) {
            this.from = from;
            this.to = to;
            this.data = data;
            this.queue = queue;
            this.ensemble = ensemble;
        }

        @Override
        protected void compute() {
            // there is no more than 5% of elements in this branch, therefore process immediately
            if ((to - from) <= ensemble.size() * EXECUTION_CUT_OFF) {
                for (int m = from; m <= to; m++) {
                    queue.add(ensemble.get(m).predict(data));
                }
            }
            // recursively divide computation until there is less there are 5% models to process in current branch
            else {
                int mid = from + (to - from) / 2;
                ModelEvaluation left = new ModelEvaluation(from, mid, data, queue, ensemble);
                ModelEvaluation right = new ModelEvaluation(mid + 1, to, data, queue, ensemble);
                left.fork();
                right.fork();
                left.join();
                right.join();
            }
        }
    }

}
