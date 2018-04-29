package ensemble;

import ensemble.model.Model;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class implements majority based voting classification and regression. This is the most democratic approach to voting
 * where each classifier, be it weak or strong, has same influence on final decision.
 *
 * <p>
 * The class offers two methods, {@link #classify(double[])} which returns most voted class given ensemble model of
 * classifiers and {@link #estimate(double[])} which estimates explanatory variable as average of ensemble model classifiers.
 * </p>
 *
 * @author dtemraz
 */
class CommitteeOfExperts {

    private final List<Model> ensembleModel;  // ensemble classifier where each classifier is equally important

     CommitteeOfExperts(List<Model> ensembleModel) {
        this.ensembleModel = ensembleModel;
    }

    /**
     * Returns expected class for <em>data</em> by majority of votes in ensemble model.
     *
     * @param data to classify
     * @return expected class for <em>data</em> by majority of votes in ensemble model
     */
    double classify(double[] data) {
        return vote(data);
    }

    /**
     * Returns regression of target value for <em>data</em> by averaging of ensemble model outputs.
     *
     * @param data for which to estimate target value
     * @return regression of target value for <em>data</em> by averaging of ensemble model outputs
     */
    double estimate(double[] data) {
        return ensembleModel.stream().mapToDouble(m -> m.predict(data))
                .reduce(Double::sum)
                .getAsDouble() / ensembleModel.size();
    }

    // each model in ensemble votes for data class and the class with most votes is chosen as target class
    private double vote(double[] data) {
        // let each model vote for a class and count votes per class
        HashMap<Double, Integer> votingResults = new HashMap<>();
        ensembleModel.forEach(model -> votingResults.merge(model.predict(data), 1, (old, n) -> old + n));

        // find class for which majority of models voted
        return votingResults.entrySet().stream()
                .max(Comparator.comparingInt(Map.Entry::getValue))
                .get()
                .getKey();
    }
}
