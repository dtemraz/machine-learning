package utilities;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author dtemraz
 */
public class ListUtilsTest {

    /**
     * Tests that lists of even size are properly split in two parts, where first part has ratio of original size and the second part has remaining items.
     */
    @Test
    public void evenSizeListShouldBeSplitInTwoParts() {
        // given
        List<Integer> list = asList(1,10);
        double splitRatio = 0.6;
        // when
        List<List<Integer>> subLists = ListUtils.split(list, splitRatio);
        // then
        assertSplit(subLists, splitRatio, list.size());
    }

    /**
     * Tests that lists of odd size are properly split in two parts, where first part has ratio of original size and the second part has remaining items.
     */
    @Test
    public void oddSizeListShouldBeSplitInTwoParts() {
        // given
        List<Integer> list = asList(1,9);
        double splitRatio = 0.6;
        // when
        List<List<Integer>> subLists = ListUtils.split(list, splitRatio);
        // then
        assertSplit(subLists, splitRatio, list.size());
    }

    /**
     * Tests that list is properly split in two parts, where first part has ratio of original size and the second part has remaining items
     * with various combinations of split ratio.
     */
    @Test
    public void listShouldBeSplitInTwoPartsWithVariousRatios() {
        // given
        List<Integer> list = asList(1, 9);
        for (double splitRatio = 0.1; splitRatio <= 1; splitRatio += 0.1) {
            // when
            List<List<Integer>> subLists = ListUtils.split(list, splitRatio);
            // then
            assertSplit(subLists, splitRatio, list.size());
        }
    }

    /**
     * Tests that list is properly partitioned in n parts where each part is of equal size. One item should be discarded since 10 is not perfectly divisible
     * over 3 partitions.
     */
    @Test
    public void evenSizeListShouldBePartitioned() {
        // given
        List<Integer> list = asList(1,10);
        int partitions = 3;
        // when
        List<List<Integer>> subLists = ListUtils.equalPartitions(list, partitions);
        // then
        int totalSize = subLists.stream().mapToInt(List::size).reduce(Integer::sum).getAsInt();
        // we should discard 1 item out of 10 to generate 3 equal sub lists, each having 3 items, which is total of 9
        assertEquals(9, totalSize);
        assertPartitions(subLists, partitions, list.size());
    }

    /**
     * Tests that list is properly partitioned in n parts where each part is of equal size. No items should be discarded since 9
     * is perfectly divisible over 3 partitions.
     */
    @Test
    public void oddSizeListShouldBePartitioned() {
        // given
        List<Integer> list = asList(1,9);
        int partitions = 3;
        // when
        List<List<Integer>> subLists = ListUtils.equalPartitions(list, partitions);
        // then
        int totalSize = subLists.stream().mapToInt(List::size).reduce(Integer::sum).getAsInt();
        // there are 9 items, which is divisible by 3 so the total size should be equal to list size
        assertEquals(list.size(), totalSize);
        assertPartitions(subLists, partitions, list.size());
    }

    private <T> void assertSplit(List<List<T>> items, double ratio, int originalSize) {
        // there should be exactly two parts
        assertEquals(2, items.size());
        // sum of each part should be equal to total number of items
        assertEquals(originalSize, items.stream().mapToInt(List::size).reduce(Integer::sum).getAsInt());
        int firstPartSize = (int)(originalSize * ratio);
        // first part's size should be ratio of original size
        assertEquals(firstPartSize, items.get(0).size());
        int secondPartSize = originalSize - firstPartSize;
        assertEquals(secondPartSize, items.get(1).size());
    }


    private <T> void assertPartitions(List<List<T>> items, int n, int originalSize) {
        // there should be n partitions
        assertEquals(n, items.size());
        // sum of each part should be less or equal to total number of items
        assertTrue(items.stream().mapToInt(List::size).reduce(Integer::sum).getAsInt() <= originalSize);
        // size of each partition should be same and equal to fraction n over original size
        int partitionSize = originalSize / n;
        items.forEach(partition -> assertEquals(partitionSize, partition.size()));
    }

    private static List<Integer> asList(Integer min, Integer max) {
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = min; i <= max; i++) {
            list.add(i);
        }
        return list;
    }

}
