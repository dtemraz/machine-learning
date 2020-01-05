package algorithms.knn;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DistanceFunctionTest {

    private static final double delta = 0.0000000001;
    private final double[] v1 = new double[] {1,1};
    private final double[] v2 = new double[] {3,3};

    @Test
    public void euclideanDistanceTest() {
        //  given
        DistanceFunction dF = DistanceFunction.EUCLIDEAN;
        // when
        double distance = dF.apply(v1, v2);
        // then
        assertEquals(Math.sqrt(8), distance,  delta);
    }

    @Test
    public void squaredEuclideanDistanceTest() {
        //  given
        DistanceFunction dF = DistanceFunction.SQUARED_EUCLIDEAN;
        // when
        double distance = dF.apply(v1, v2);
        // then
        assertEquals(8, distance,  0);
    }


    @Test
    public void manhattanDistanceTest() {
        //  given
        DistanceFunction dF = DistanceFunction.MANHATTAN;
        // when
        double distance = dF.apply(v1, v2);
        // then
        assertEquals(4, distance,  0);
    }


}
