package algorithms.k_means;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * This class implements K-Means clustering algorithm. The user should provide initial centroids via constructor {@link #K_Means(List)}},
 * and the algorithm will create number of clusters equal to number of centroids.
 * <p>
 * The user can cluster data with method {@link #cluster(List)} which will stop within at most {@link #MAX_ITERATIONS} and
 * print all getMembers of a given cluster with method {@link #membersFor(int)}.
 * </p>
 *
 * @author dtemraz
 */
public class K_Means {

    private static final int MAX_ITERATIONS = 10_000;

    private final Cluster[] clusters;

    /**
     * Constructs {@link K_Means} instance with K equal to number of elements in <em>centroids</em>.
     *
     * @param centroids which define initial clusters
     */
    public K_Means(List<Double[]> centroids) {
        // centroids define initial clusters
        clusters = new Cluster[centroids.size()];
        for (int centroid = 0; centroid < centroids.size(); centroid++) {
            clusters[centroid] = new Cluster(new Member(centroids.get(centroid)));
        }
    }

    /**
     * Clusters <em>data</em> into clusters defined with initial centroids.
     *
     * @param data to cluster
     */
    public void cluster(List<Double[]> data) {
        data.forEach(this::mergeWithClosest);
        // some data samples which came early could end up in a wrong cluster since cluster centroids might have shifted
        // quite a bit after these samples were added
        for (int balanceIteration = 0; balanceIteration < MAX_ITERATIONS; balanceIteration++) {
            if (balance()) { // this will fix clustering for wrongly clustered samples, if there are any
                break;
            }
        }
    }

    /**
     * Returns all cluster members for a given cluster defined with <em>clusterId</em> which could be any value between 0 and K-1.
     *
     * @param clusterId any value between 0 and K-1
     * @return all cluster members for a given cluster defined with <em>clusterId</em> which could be any value between 0 and K-1
     */
    public Collection<Member> membersFor(int clusterId) {
        return clusters[clusterId].getMembers();
    }

    // adds data into closest cluster
    private void mergeWithClosest(Double[] data) {
        minDistanceCentroid(data).add(new Member(data));
    }

    // returns closest cluster for data
    private Cluster minDistanceCentroid(Double[] data) {
        Cluster minCentroid = null;
        double minDistance = Double.MAX_VALUE;
        for (Cluster cluster : clusters) {
            double clusterDistance = cluster.distanceTo(data);
            if (clusterDistance < minDistance) {
                minCentroid = cluster;
                minDistance = clusterDistance;
            }
        }
        return minCentroid;
    }

    // moves wrongly clustered members to correct clusters, in which case method returns true; returns false if all
    // samples were correctly clustered
    private boolean balance() {
        Map<Cluster, List<Double[]>> wronglyClassified = new HashMap<>();
        for (Cluster current : clusters) {
                Predicate<Double[]> obsolete = data -> {
                    Cluster closest = minDistanceCentroid(data); // check if there is a cluster that is closer to data
                    if (!current.equals(closest)) {
                        // save map of target clusters and data that should be added to them
                        wronglyClassified.merge(closest, asList(data), (old, n) -> {old.addAll(n); return old;});
                        return true;
                    }
                    return false;
                };
            fixClustering(current, wronglyClassified, obsolete);
        }
        return wronglyClassified.size() == 0; // we didn't move any samples between clusters
    }

    // removes data from cluster which does not belong to it and adds the data to closer cluster
    private void fixClustering(Cluster cluster, Map<Cluster, List<Double[]>> wronglyClassified, Predicate<Double[]> obsolete) {
        // both calls bellow could disturb cluster balance for other samples, therefore we need iterative refinement
        cluster.remove(obsolete);
        moveToTargetCluster(wronglyClassified);

        wronglyClassified.clear();
    }

    // moves list of data to their associated cluster
    private void moveToTargetCluster(Map<Cluster, List<Double[]>> wronglyClassified) {
        for (Map.Entry<Cluster, List<Double[]>> entry : wronglyClassified.entrySet()) {
            Cluster targetCluster = entry.getKey();
            for (Double[] data : entry.getValue()) {
                targetCluster.add(new Member(data));
            }
        }
    }

    private List<Double[]> asList(Double[] item) {
        ArrayList<Double[]> list = new ArrayList<>();
        list.add(item);
        return list;
    }

}
