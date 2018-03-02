package structures;

import utilities.Vector;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * This class implements randomized queue structure where item dequeue order is not deterministic. The user is able to add an item
 * to the queue with method {@link #enqueue(Object)} and dequeue an item in a non deterministic order with method {@link #dequeue()}.
 * The class also exposes method {@link #sample()} which returns random sample from this dequeue without actually removing the item.
 * There are also standard {@link #isEmpty()} and {@link #size()} methods.
 * <p>
 * Finally, user can iterate this queue in random order without removal of items. Each {@link #iterator()} has a uniform chance
 * to return any permutation of items in this queue, therefore it is unlikely that two iterators over same dequeue give
 * same item traversal order.
 * </p>
 *
 * All operations besides {@link #iterator()} are guaranteed to run in constant (amortized) time.
 *
 * @author dtemraz
 * 
 * @param <Item> type of item stored in queue
 */
public class RandomizedQueue<Item> implements Iterable<Item> {
    
    public static final int DEFAULT_CAPACITY = 8;

    private final Random random;
    private int size;
    private Item[] items;
    
    public RandomizedQueue() {
        items = createEmptyItems(DEFAULT_CAPACITY);
        random = new Random();
    }

    /**
     * Returns true if this queue contains no elements, false otherwise.
     * 
     * @return true if this queue contains no elements, false otherwise
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Returns number of the items in this queue.
     * 
     * @return number of items in this queue
     */
    public int size() {
        return size;
    }

    /**
     * Adds the item to the queue.
     * 
     * @param item to add into queue
     */
    public void enqueue(Item item) {
        if (item == null) {
            throw new NullPointerException("item must not be null");
        }
        // double the capacity if the queue is full
        if (isFull()) {
            resize(size * 2);
        }
        items[size++] = item;
    }

    /**
     * Removes and returns a random item from the queue.
     * 
     * @return random item from the queue
     */
    public Item dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException("cannot take an item from the empty queue");
        }
        // takes random item from this queue and swaps it's position with last item, this ensures constant time deque when
        // last item is removed
        Item item = pickRandom();
        items[size] = null;
        if (isQuarterFull()) {
            resize(items.length / 2);
        }
        return item;
    }

    /**
     * Returns random item from the queue, without removing it.
     * 
     * @return random item from the queue, without removing it
     */
    public Item sample() {
        if (isEmpty()) {
            throw new NoSuchElementException("cannot take an item from the empty queue");
        }
        return items[sampleIndex()];
    }
    
    /**
     * Returns an iterator over the random permutation of this queue.
     * Remove operation is not supported.
     * 
     * @return non-deterministic order iterator
     */
    @Override
    public Iterator<Item> iterator() {
        // permute new items instance to preserve insertion order per assignment requirements
        Item[] permutation = createEmptyItems(size);
        copyItems(permutation);
        // pick uniformly a permutation of items
        Vector.shuffle(permutation);
        return iterator(permutation);
    }
    
    private Item pickRandom() {
        // pick random sample from this queue
        int sample = sampleIndex();
        Item item = items[sample];
        // if non-last item was picked move last item into position of the sample
        if (sample < --size) {
            items[sample] = items[size];
        }
        return item;
    }

    private int sampleIndex() {
        return random.nextInt(size);
    }

    // array maintenance

    private boolean isFull(){
        return size == items.length;
    }
    
    private boolean isQuarterFull() {
        final int quarterFull = 4;
        return size > 0 && items.length / size == quarterFull;
    }

    private void resize(int capacity) {
        Item[] resized = createEmptyItems(capacity);
        copyItems(resized);
        items = resized;
    }

    private void copyItems(Item[] dest) {
        System.arraycopy(items, 0, dest, 0, size);
    }

    /*
     * Java doesn't support creation on generic arrays hence this cast is inevitable.
     * Although it cannot be proven by compiler, the cast is perfectly safe since items[] reference
     * is private and never leaked, therefore client is unable to put anything other than the item instance into this queue.
     */
    @SuppressWarnings("unchecked")
    private Item[] createEmptyItems(int capacity) {
        return (Item[]) new Object[capacity];
    }

    
    // returns iterator over the items, does not support remove operation
    private Iterator<Item> iterator(Item[] items) {
        return new Iterator<Item>() {
            int current = 0;

            @Override
            public boolean hasNext() {
                return current < items.length;
            }

            @Override
            public Item next() {
                if (!hasNext()) {
                    throw new NoSuchElementException("no more elements in this iteration");
                }
                return items[current++];
            }

            /**
             * This operation is not supported and will result with {@link UnsupportedOperationException} exception.
             *
             * @throws UnsupportedOperationException if remove is invoked
             */
            @Override
            public void remove() {
                throw new UnsupportedOperationException("remove operation not supported");
            }
        };
    }
    
}