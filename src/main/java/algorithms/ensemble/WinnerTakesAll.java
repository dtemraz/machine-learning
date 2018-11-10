package algorithms.ensemble;

import algorithms.model.ClassificationResult;
import algorithms.model.TextModel;
import algorithms.model.TextModelWithProbability;

import java.util.List;

/**
 * This class implements simple arbitrator strategy when there are multiple independent {@link TextModelWithProbability} implementations.
 * The class simply selects the model with highest confidence score as the winner.
 * <p>
 * The alternate and more sophisticated strategy would be to train a model on responses from wrapped models.
 * </p>
 *
 * @author dtemraz
 */
public class WinnerTakesAll implements TextModel {

    private final TextModelWithProbability[] textModels; // independent models which can output confidence score with their classification

    public WinnerTakesAll(List<TextModelWithProbability> textModels) {
        // array iterator is ˇ~30% cheaper than the list iterator
        this.textModels = new TextModelWithProbability[textModels.size()];
        for (int model = 0; model < textModels.size(); model++) {
            this.textModels[model] = textModels.get(model);
        }
    }

    @Override
    public double classify(String[] words) {
        double classId = -1;
        double maxProb = Double.NEGATIVE_INFINITY;
        // find max -> select the class from model with highest confidence score
        for (TextModelWithProbability textModel : textModels) {
            ClassificationResult classificationResult = textModel.classifyWithProb(words);
            double probability = classificationResult.getProbability();
            if (probability > maxProb) {
                maxProb = probability;
                classId = classificationResult.getClassId();
            }
        }
        return classId;
    }
}
