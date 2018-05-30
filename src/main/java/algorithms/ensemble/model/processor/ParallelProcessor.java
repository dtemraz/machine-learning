package algorithms.ensemble.model.processor;

import algorithms.ensemble.model.Model;
import algorithms.ensemble.model.TextModel;

import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * This class can be used to satisfy {@link EnsembleModelProcessor#predictions(List, double[])} with parallel evaluation
 * of models.
 *
 * @author dtemraz
 */
public class ParallelProcessor {

    // could have been in lambda body, but lazy initialization is slow
    private static final ForkJoinPool executor = new ForkJoinPool();

    // no need to instantiate this class
    private ParallelProcessor() { }


    /**
     * Returns prediction for each model in algorithms.ensemble in parallel mode
     *
     * @see EnsembleModelProcessor#predictions(List, double[])
     */
    public static double[] predictions(List<Model> ensemble, double[] data) {
        ConcurrentLinkedQueue<Double> predictions = new ConcurrentLinkedQueue<>();
        executor.submit(new ModelEvaluation(0, ensemble.size() - 1, data, predictions, ensemble)).join();
        return collectResults(predictions, ensemble.size());
    }

    public static double[] predictions(List<TextModel> ensemble, String[] data) {
        ConcurrentLinkedQueue<Double> predictions = new ConcurrentLinkedQueue<>();
        executor.submit(new TextModelEvaluation(0, ensemble.size() - 1, data, predictions, ensemble)).join();
        return collectResults(predictions, ensemble.size());
    }

    private static double[] collectResults(ConcurrentLinkedQueue<Double> predictions, int size) {
        double[] results = new double[size];
        for (int prediction = 0; prediction < results.length; prediction++) {
            results[prediction] = predictions.poll();
        }
        return results;
    }


    /**
     * The class implements parallel algorithms.ensemble model evaluation with the work stealing semantics. Maybe a bit unintuitive,
     * but the class it not an instance of {@link java.util.concurrent.RecursiveTask}. It could have merged different array
     * pieces into a final array which would be returned, but it doesn't because reasons.
     *
     * Prediction of each model is stored in concurrent queue which can be evaluated once the computation finishes.
     */
    private static class ModelEvaluation extends RecursiveAction {

        private static final double EXECUTION_CUT_OFF = 0.05;

        private final int from; // first model that should be evaluated by this task
        private final int to; // last model that should be evaluated by this task
        private final double[] data; // data for which to make prediction, constant between instances
        private final ConcurrentLinkedQueue<Double> queue; // stores prediction of each model, constant reference

        private final List<Model> ensemble; // set of models that should be evaluated, constant between instances

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
            // recursively divide computation until there is no more than 5% models to process in current branch
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

    /**
     * Disclaimer: this class is a shameless copy past modified to work with String[] for the simple reason that there is no
     * common way to express array of primitives(double[]) and array of Strings in Java via.
     * There could have been generic {@link Model} which could be parametrized with wrapper type Double but i preferred
     * a little bit of copy paste to retain primitives.
     */
    private static class TextModelEvaluation extends RecursiveAction {

        private static final double EXECUTION_CUT_OFF = 0.05;

        private final int from; // first model that should be evaluated by this task
        private final int to; // last model that should be evaluated by this task
        private final String[] data; // data for which to make prediction, constant between instances
        private final ConcurrentLinkedQueue<Double> queue; // stores prediction of each model, constant reference

        private final List<TextModel> ensemble; // set of models that should be evaluated, constant between instances

        private TextModelEvaluation(int from, int to, String[] data, ConcurrentLinkedQueue<Double> queue, List<TextModel> ensemble) {
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
                    queue.add(ensemble.get(m).classify(data));
                }
            }
            // recursively divide computation until there is no more than 5% models to process in current branch
            else {
                int mid = from + (to - from) / 2;
                TextModelEvaluation left = new TextModelEvaluation(from, mid, data, queue, ensemble);
                TextModelEvaluation right = new TextModelEvaluation(mid + 1, to, data, queue, ensemble);
                left.fork();
                right.fork();
                left.join();
                right.join();
            }
        }
    }

}
