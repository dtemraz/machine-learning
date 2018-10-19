package algorithms.ensemble;

import algorithms.ensemble.model.Model;
import algorithms.ensemble.model.ModelSupplier;
import algorithms.ensemble.model.TextModel;
import algorithms.ensemble.model.processor.ParallelProcessor;
import algorithms.ensemble.model.processor.TextEnsembleModelProcessor;
import structures.text.Vocabulary;
import structures.text.TF_IDF_Vectorizer;
import structures.text.SparseDenseSample;
import utilities.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * This class enables stacked generalization of {@link TextModel}s with combiner as instance of {@link Model}.
 *
 * <p>
 * The implementation assumes that level-0 models will be trained with sparse features and level-1 combiner will be trained
 * with dense features and outputs from level-0.
 * </p>
 *
 * @see StackedGeneralization
 *
 * @author dtemraz
 */
public class TextModelsStacking {

    private final List<TextModel> models; // level-0 text models used for prediction over sparse features
    private final TextEnsembleModelProcessor processor; // sequential/parallel execution of level-0 models
    private final Vocabulary denseVocabulary; // vocabulary of allowed dense features
    private final Model combiner; // supervised level-1 model used to combine outputs from level-1 models for final results

    /**
     * Builds instance of {@link TextModelsStacking} used to combine outputs from <em>models</em> in supervised manner.
     * The <em>trainingSet</em> contains sparse features which shall be used to train level-0 models and dense features
     * to train the level-1 model together with the output from level-0.
     * <p>
     * Combiner is defined with <em>combinerSupplier</em> implementation and the invocation of supplier should provide instance
     * which can be trained given the outputs and vectorized dense features in form of <em>double[]</em>.
     * </p>
     * Finally, <em>denseVocabulary</em> is used to extract features from dense textual features.
     *
     * @param models text models which form together level-0 predictor
     * @param trainingSet with sparse and dense features per class
     * @param combinerSupplier that should return trainable instance of a combiner {@link Model}
     * @param denseVocabulary
     */
    public TextModelsStacking(List<TextModel> models, List<SparseDenseSample> trainingSet, ModelSupplier combinerSupplier,
                              Vocabulary denseVocabulary) {
        this.models = models;
        this.processor = ParallelProcessor::predictions;
        this.denseVocabulary = denseVocabulary;
        this.combiner = combinerSupplier.get(combinerTrainingSet(trainingSet));
    }

    /**
     * Returns predicted class given the sparse and dense features of a text to classify.
     *
     * @param sparse set of 'rare' features, such as TF-IDF in short texts over large corpus
     * @param dense set of common features, such as flags like has alphanumeric words, currency signs and so on
     * @return predicted class for sparse and dense features of a text
     */
    public double predict(String[] sparse, String[] dense) {
        // make a level-0 prediction over sparse features
        double[] predictions = processor.predictions(models, sparse);
        double[] denseFeatures = TF_IDF_Vectorizer.vectorize(dense, denseVocabulary);
        // aggregate prediction results and dense features into final feature set for combiner
        return combiner.predict(Vector.merge(denseFeatures, predictions));
    }

    /**
     * Returns training set for a combiner constituted of predictions from level-0 models, dense features and target class.
     *
     * @param samples from which to train level-0 models with sparse features, and level-1 model with dense features
     * @return training set for a combiner constituted of predictions from level-0 models, dense features and target class
     */
    private List<double[]> combinerTrainingSet(List<SparseDenseSample> samples) {
        List<double[]> combinerSet = new ArrayList<>(); // level-0 predictions, dense features and target class
        for (SparseDenseSample sparseDenseSample : samples) {
            List<String[]> sparse = sparseDenseSample.getSparse(); // train level-0 models
            List<String[]> dense = sparseDenseSample.getDense(); // train combiner
            for (int sample = 0; sample < sparse.size(); sample++) {
                double[] level0Predictions = processor.predictions(models, sparse.get(sample));
                // construct rows which contain predictions for sparse data and the target class
                double[] predictionsWithClass = Vector.merge(level0Predictions, sparseDenseSample.getClassId());
                // vectorize dense features with the respective vocabulary
                double[] denseFeatures = TF_IDF_Vectorizer.vectorize(dense.get(sample), denseVocabulary);
                // merge dense features with prediction set as a final learning vector for combiner
                combinerSet.add(Vector.merge(denseFeatures, predictionsWithClass));
            }
        }
        return combinerSet;
    }

}