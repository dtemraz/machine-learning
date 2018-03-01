package examples.clustering;

import unsupervised.clustering.K_Means;

import java.util.ArrayList;
import java.util.List;

/**
 * @author dtemraz
 */
public class TestClustering {

    public static void main(String[] args) {
        List<Double[]> c = new ArrayList<>();
        c.add(new Double[]{1.0,1.0});
        c.add(new Double[]{5.0,7.0});
        K_Means k = new K_Means(c);

        List<Double[]> d = new ArrayList<>();
        d.add(new Double[]{1.5, 2.0});
        d.add(new Double[]{3.0, 4.0});
        d.add(new Double[]{3.5, 5.0});
        d.add(new Double[]{4.5, 5.0});
        d.add(new Double[]{3.5, 4.0});
        k.cluster(d);

        k.membersFor(0).forEach(System.out::println); // expected -> {1.0, 1.0}, {1.5, 2.0}
        System.out.println();
        k.membersFor(1).forEach(System.out::println); // expected -> {5.0, 7.0}, {3.5, 5.0}, {4.5, 5.0}, {3.5, 4.0}

    }

}
