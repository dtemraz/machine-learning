package utilities;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This is a utility class which lets user split lists, either in two parts with some ratio or n equal parts where remaining elements are discarded.
 * <p>
 * Methods {@link #split(List, double)} and {@link #randomizedSplit(List, double)} (List, double)} split the list in two parts, former also shuffles
 * lists before split. The user may also split the list into <em>n</em> equal partitions with the methods {@link #equalPartitions(List, int)} and {@link #randomizedEqualPartitions(List, int)},
 * the former again performs shuffle before partitioning the list.
 * </p>
 * @author dtemraz
 */
public class ListUtils {

    /**
     * Returns list split <em>items</em> in two parts where first part will have <em>ratio</em> of size elements and the second part will have remaining items.
     * The method runs in constant time and returns views of the <em>items</em> list since it is backed with {@link List#subList(int, int)}.
     * <p>
     * This method additionally performs shuffle of <em>items</em> before it splits the list.
     * </p>
     *
     * @param items to split according to ratio
     * @param ratio which defines size of the <strong>first</strong> part, second part will have remaining items
     * @param <T> type of item
     * @return list split <em>items</em> in two parts where first part will have <em>ratio</em> of size elements and the second part will have remaining items
     */
    public static <T> List<List<T>> randomizedSplit(List<T> items, double ratio) {
        Collections.shuffle(items);
        return split(items, ratio);
    }

    /**
     * Returns list split <em>items</em> in two parts where first part will have <em>ratio</em> of size elements and the second part will have remaining items.
     * The method runs in constant time and returns views of the <em>items</em> list since it is backed with {@link List#subList(int, int)}.
     *
     * @param items to split according to ratio
     * @param ratio which defines size of the <strong>first</strong> part, second part will have remaining items
     * @param <T> type of item
     * @return list split <em>items</em> in two parts where first part will have <em>ratio</em> of size elements and the second part will have remaining items
     */
    public static <T> List<List<T>> split(List<T> items, double ratio) {
        int splitSize = (int) (items.size() * ratio);
        List<List<T>> subLists = new ArrayList<>();
        subLists.add(items.subList(0, splitSize));
        subLists.add(items.subList(splitSize, items.size()));
        return subLists;
    }

    /**
     * Returns <em>items</em> list split into <em>n</em> <strong>equal</strong> partitions. Remainder elements are <strong>discarded</strong>.
     * The method runs in constant time(technically in time proportional to number of partitions <em>n</em>) and returns views of the <em>items</em> list
     * since it is backed with {@link List#subList(int, int)}.
     * <p>
     * If <em>items</em> are of size 10, and <em>n=3</em>, the method will split the list into 3 sub-lists, each having 3 items and discarding 10th item.
     * </p>
     * This method additionally performs shuffle of <em>items</em> before it splits the list.
     *
     * @param items to split into n <strong>equal</strong> partitions
     * @param n number of partitions
     * @param <T> type of items
     * @return <em>items</em> list split into <em>n</em> <strong>equal</strong> partitions. Remainder elements are <strong>discarded</strong>
     */
    public static <T> List<List<T>> randomizedEqualPartitions(List<T> items, int n) {
        Collections.shuffle(items);
        return equalPartitions(items, n);
    }

    /**
     * Returns <em>items</em> list split into <em>n</em> <strong>equal</strong> partitions. Remainder elements are <strong>discarded</strong>.
     * The method runs in constant time(technically in time proportional to number of partitions <em>n</em>) and returns views of the <em>items</em> list
     * since it is backed with {@link List#subList(int, int)}.
     * <p>
     * If <em>items</em> are of size 10, and <em>n=3</em>, the method will split the list into 3 sub-lists, each having 3 items and discarding 10th item.
     * </p>
     *
     * @param items to split into n <strong>equal</strong> partitions
     * @param n number of partitions
     * @param <T> type of items
     * @return <em>items</em> list split into <em>n</em> <strong>equal</strong> partitions. Remainder elements are <strong>discarded</strong>
     */
    public static <T> List<List<T>> equalPartitions(List<T> items, int n) {
        List<List<T>> subLists = new ArrayList<>();
        int partitionSize = items.size() / n;
        for (int partition = 0; partition < n; partition++) {
            int from = partition * partitionSize;
            int to = from + partitionSize;
            subLists.add(items.subList(from, to));
        }
        return subLists;
    }
}
