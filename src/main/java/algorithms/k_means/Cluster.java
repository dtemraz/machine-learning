package algorithms.k_means;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Predicate;

/**
 * This class implements cluster data type used in {@link K_Means} algorithm. The cluster should receive initial centroid
 * via constructor {@link #Cluster(Member)}.
 *
 * <p>
 * The class offers method {@link #add(Member)} to add a new member to cluster, {@link #remove(Predicate)} to remove all members
 * that meet removal criteria, {@link #distanceTo(Double[])} to calculate squared euclidean distance for a data sample
 * to cluster instance,
 * {@link #size()} which returns number of cluster members and {@link #getMembers()} which returns all cluster members.
 * </p>
 *
 * @author dtemraz
 */
public class Cluster {

    private final Member centroid; // mean value of cluster
    private final Double[] sum; // by preserving sum of all cluster members we can easily compute new centroid when data is added to cluster
    private final List<Member> members; // members belonging to this cluster

    // using squared euclidean since root operation might end up being bottleneck
    private static final BiFunction<Double[], Double[], Double> squaredEuclidean = (a, b) -> {
        int length = a.length; // could be either a or b
        double distance = 0;
        // this is just euclidean distance
        for (int component = 0; component < length; component++) {
            double delta = a[component] - b[component];
            distance += delta * delta;
        }
        return distance * distance;
    };

    /**
     * Constructs {@link Cluster} instance with initial <em>centroid</em>
     *
     * @param centroid initial centroid
     */
    public Cluster(Member centroid) {
        this.centroid = centroid;
        this.members = new ArrayList<>();
        Double[] data = centroid.data();
        // we gonna modify centroid in algorithm, therefore we need a deep copy
        this.sum = Arrays.copyOf(data, data.length);
        members.add(new Member(Arrays.copyOf(data, data.length)));
    }

    /**
     * Adds <em>member</em> to this cluster.
     *
     * @param member to add into cluster
     */
    public void add(Member member) {
        members.add(member);
        updateCentroid(member.data(), Double::sum);
    }

    /**
     * Removes all cluster members which satisfy <em>removeCondition</em> and returns list of removed members,
     * empty list if no members satisfy condition.
     *
     * @param removeCondition which a member should satisfy to be removed
     * @return list of removed members, empty list if no members satisfy condition.
     */
    public List<Double[]> remove(Predicate<Double[]> removeCondition) {
        List<Double[]> removed = new ArrayList<>(); // maintain list of removed members so we can return it
        Iterator<Member> clusterIterator = members.iterator();
        while (clusterIterator.hasNext()) {
            Double[] data = clusterIterator.next().data();
            if (removeCondition.test(data)) {
                clusterIterator.remove();
                removed.add(data);
                updateCentroid(data, (a, b) -> a - b); // update centroid for each removed member
            }
        }
        return removed;
    }

    /**
     * Returns squared euclidean distance of this cluster to <em>data</em>.
     *
     * @param data for which to calculate distance to this cluster
     * @return squared euclidean distance of this cluster to <em>data</em>
     */
    public double distanceTo(Double[] data) {
        return squaredEuclidean.apply(data, centroid.data());
    }

    /**
     * Returns number of members in this cluster.
     *
     * @return number of members in this cluster
     */
    public int size() {
        return members.size();
    }

    /**
     * Returns an unmodifiable view of cluster members.
     *
     * @return unmodifiable view of cluster members.
     */
    public Collection<Member> getMembers() {
        return Collections.unmodifiableCollection(members);
    }

    /**
     * Returns mean of all cluster members.
     *
     * @return mean of all cluster members
     */
    public Double[] getCentroid() {
        Double[] centroidValue = centroid.data();
        return Arrays.copyOf(centroidValue, centroidValue.length);
    }

    @Override
    public String toString() {
        StringJoiner joiner = new StringJoiner(",");
        members.forEach(m -> joiner.add(m.toString()));
        return joiner.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Cluster cluster = (Cluster) o;
        return centroid.equals(cluster.centroid);
    }

    @Override
    public int hashCode() {
        return centroid.hashCode();
    }

    // updates centroid with data according to operator which should be either summation or subtraction operator
    private void updateCentroid(Double[] data, BinaryOperator<Double> operator) {
        Double[] centroidValue = centroid.data();
        // since we are maintaining sum, no need to iterate all previous cluster members to calculate new centroid
        for (int component = 0; component < data.length; component++) {
            sum[component] = operator.apply(sum[component], data[component]);
            centroidValue[component] = sum[component] / size();
        }
    }

}
