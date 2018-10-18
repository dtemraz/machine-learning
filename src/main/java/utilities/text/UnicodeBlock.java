package utilities.text;

/**
 * This class represents any Unicode block and may answer if a given character belongs to that block.
 *
 * @author dtemraz
 */
public class UnicodeBlock {

    private final int low;
    private final int high;

    public UnicodeBlock(int low, int high) {
        if (high <= low) {
            throw new IllegalArgumentException("high must be greater than low");
        }
        this.low = low;
        this.high = high;
    }

    /**
     * Returns <em>true</em> if the character <em>c</em> belongs to this Unicode block, <em>false</em> otherwise.
     *
     * @param c character to check if it belongs to this block
     * @return <em>true</em> if the character <em>c</em> belongs to this Unicode block, <em>false</em> otherwise
     */
    public boolean contains(char c) {
        if (c >= low && c <= high) {
            return true;
        }
        return false;
    }

}
