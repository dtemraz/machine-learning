package algorithms.knn;

/**
 * This enum lets user setup KNN for classification or regression in weighted or non-weighted form.
 *
 * There are five strategies to chose from:
 * <ul>
 * <li>{@link #AVERAGE} used in <em>regression</em>, returns average value of all K neighbors regardless of their distance</li>
 *
 * <li>{@link #INVERSE_DISTANCE_WEIGHTED_MAJORITY} used in <em>regression</em>, returns average value of all K neighbors
 * giving more importance to closer neighbors</li>
 *
 * <li>{@link #MAJORITY} used in <em>classifications</em>, returns class which corresponds to most of the K neighbors
 * regardless of their distance to the query point</li>
 *
 * <li>{@link #INVERSE_DISTANCE_WEIGHTED_MAJORITY} used in <em>classifications</em>, returns class which corresponds to most
 * of the K neighbors giving more importance to closer neighbors</li>
 *
 * <li>{@link #INVERSE_CLASS_PROBABILITY_WEIGHTED_MAJORITY} used in <em>classifications</em>, returns class which corresponds to most
 * of the K neighbors weighting each neighbor by the inverse of probability of its associated class.
 * Intuitively if a class more common it should appear more often near query point</li>
 * </ul>
 *
 * @author dtemraz
 */
public enum AggregationStrategy {
    // regression
    AVERAGE,
    INVERSE_DISTANCE_WEIGHTED_AVERAGE,

    // classification
    MAJORITY,
    INVERSE_DISTANCE_WEIGHTED_MAJORITY,
    INVERSE_CLASS_PROBABILITY_WEIGHTED_MAJORITY
}
