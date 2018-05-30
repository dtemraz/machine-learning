package algorithms.ensemble.model;

import java.io.Serializable;

/**
 * This interface is modification of {@link Model} specialized to work with Strings. It's pretty much copy-paste idea since
 * there is no way to express generic model which accepts primitive array and string array.
 *
 * @author dtemraz
 */
@FunctionalInterface
public interface TextModel extends Serializable {

    /**
     * Returns predicted class for the <em>words</em>.
     *
     * @param words to classify
     * @return classification for <em>words</em>
     */
    double classify(String[] words);
}
