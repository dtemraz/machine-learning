package algorithms.knn;

import utilities.QuickSelect;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * This interface lets user specify nearest neighbor search strategy for KNN algorithm.
 * There are two possible implementations:
 * <ul>
 *     <li>sorting with linearithmic complexity</li>
 *     <li>quick select with linear complexity</li>
 * </ul>
 * QuickSelect version is the default one, it should outperform sorting variant in most if not all realistic scenarios.
 */
public interface SearchStrategy extends Serializable {

    /**
     * Returns list of K distances for neighbors closest to the target point.
     *
     * @param input list of calculated distances for the target point
     * @param K number of neighbor distances to consider
     * @return list of K distances for neighbors closest to the target point
     */
    List<KNearestNeighbors.Neighbor> findKNearest(List<KNearestNeighbors.Neighbor> input, int K);

    // linearithmic complexity
    SearchStrategy SORTING = (input, k) -> {
        Collections.sort(input);
        return input.subList(0, k);
    };

    // linear complexity
    SearchStrategy QUICK_SELECT = QuickSelect::kSmallest;
}
