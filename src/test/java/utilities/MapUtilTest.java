package utilities;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class MapUtilTest {

    private static final double KEY = 0D;

    private Map<Double, List<String[]>> dataSet;

    @Before
    public void setUp() {
        dataSet = new HashMap<>();
    }

    @Test
    public void numberShouldBeRemoved() {
        // given
        dataSet.put(KEY, asList(new String[]{"12345"}, new String[]{"5632"}));
        // when
        MapUtil.removeNumbers(dataSet);
        // then
        assertEquals(0, dataSet.get(KEY).size());
    }

    @Test
    public void alphanumericsShouldNotBeRemoved() {
        // given
        String alphaNumeric = "bla12345";
        dataSet.put(KEY, asList(new String[]{alphaNumeric}, new String[]{"5632"}));
        // when
        MapUtil.removeNumbers(dataSet);
        // then
        assertEquals(1, dataSet.get(KEY).size());
        assertEquals(alphaNumeric, dataSet.get(KEY).get(0)[0]);
    }

    @Test
    public void regularWordsShouldNotBeRemoved() {
        // given
        dataSet.put(KEY, asList(new String[]{"foo"}, new String[]{"bar"}));
        // when
        MapUtil.removeNumbers(dataSet);
        // then
        assertEquals(2, dataSet.get(KEY).size());
    }

    @Test
    public void singleLetterWordsShouldBeRemoved() {
        // given
        dataSet.put(KEY, asList(new String[]{"c"}, new String[]{"o"}, new String[]{"n"}, new String[]{"f"}, new String[]{"i"}, new String[]{"r"}, new String[]{"m"}, new String[]{"12345"}));
        // when
        MapUtil.removeNumbersAndSingleLetterWords(dataSet);
        // then
        assertEquals(0, dataSet.get(KEY).size());
    }

    private List<String[]> asList(String[]... items) {
        List<String[]> features = new ArrayList<>();
        for(String[] item : items) {
            features.add(item);
        }
        return features;
    }

}
