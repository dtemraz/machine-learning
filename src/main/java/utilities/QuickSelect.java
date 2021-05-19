package utilities;

import java.util.Collections;
import java.util.List;

/**
 * This class is slight modification of: https://algs4.cs.princeton.edu/23quicksort/Quick.java.html
 * which exposes additional API method to select K smallest elements {@link #kSmallest(List, int)}.
 */
public class QuickSelect {

    /**
     * Returns K smallest elements in the <em>items</em> array in <strong>non</strong> sorted order.
     * Note that this function <strong>mutates</strong> <em>items</em> list similar to {@link #select(List, int)} method.
     *
     * @param items the array
     * @param k the rank of the key
     * @param <T> type of comparable item
     * @return K smallest elements in the <em>items</em> array in <strong>non</strong> sorted order
     * @throws IllegalArgumentException if <em>k</em> is negative or greater than <em>items</em> size
     */
    public static <T extends Comparable<? super T>> List<T> kSmallest(List<T> items, int k) {
        // piggy-back on invariant of select method, all elements left of rank K item are smaller(but not sorted) than Kth item
        select(items, k);
        return items.subList(0, k + 1);
    }

    /**
     * Rearranges the array so that {@code items[k]} contains the kth smallest key;
     * {@code items[0]} through {@code items[k-1]} are less than (or equal to) {@code items[k]}; and
     * {@code items[k+1]} through {@code items[n-1]} are greater than (or equal to) {@code items[k]}.
     *
     * @param items the array
     * @param k the rank of the key
     * @param <T> type of comparable item
     * @return the key of rank {@code k}
     * @throws IllegalArgumentException unless {@code 0 <= k < items.length}
     */
    public static <T extends Comparable<? super T>> T select(List<T> items, int k) {
        if (k < 0 || k >= items.size()) {
            throw new IllegalArgumentException("index is not between 0 and " + items.size() + ": " + k);
        }
        Collections.shuffle(items);
        int lo = 0, hi = items.size() - 1;
        while (hi > lo) {
            int i = partition(items, lo, hi);
            if (i > k) { hi = i - 1; } else if (i < k) { lo = i + 1; } else { return items.get(i); }
        }
        return items.get(lo);
    }

    private static <T extends Comparable<? super T>> int partition(List<T> a, int lo, int hi) {
        int i = lo;
        int j = hi + 1;
        T v = a.get(lo);
        while (true) {
            while (less(a.get(++i), v)) {
                if (i == hi) { break; }
            }
            while (less(v, a.get(--j))) {
                if (j == lo) {
                    break;
                }
            }
            if (i >= j) { break; }
            exchange(a, i, j);
        }
        exchange(a, lo, j);
        return j;
    }

    private static <T extends Comparable<? super T>> boolean less(T a, T b) {
        if (a == b) {
            return false;
        }
        return a.compareTo(b) < 0;
    }

    private static <T extends Comparable<? super T>> void exchange(List<T> a, int i, int j) {
        T swap = a.get(i);
        a.set(i, a.get(j));
        a.set(j, swap);
    }

}
