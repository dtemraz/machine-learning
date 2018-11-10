package evaluation;

import lombok.Data;

import java.util.HashMap;
import java.util.List;

/**
 * Simple wrapper class used to contain both the training set and validation set across all classes.
 *
 * @param <T> type of data in the sets
 */
@Data
public class TrainAndTestSplit<T> {
    final HashMap<Double, List<T>> trainingSet;
    final HashMap<Double, List<T>> validationSet;
}
