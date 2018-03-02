package k_means;

import java.util.Arrays;

/**
 * This class implements member model for {@link Cluster}. The class is a thin wrapper for Double[] and aims to provide
 * convenient implementation od methods equals, hashcode and toString.
 * Note that arrays in java cannot be used as hashing keys since they inherit default implementation and do not override it.
 *
 * @author dtemraz
 */
public class Member {

    private final Double[] data;

    private int hashcode = -1;

    public Member(Double[] data) {
        this.data = data;
    }

    // expected to be used often internally, therefore it's better to not make defensive copy
    Double[] data() {
        return data;
    }

    /**
     * Returns data contained in this member.
     *
     * @return data contained in this member
     */
    public Double[] getData() {
        return Arrays.copyOf(data, data.length);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (other == null || getClass() != other.getClass()) return false;
        return Arrays.equals(data, ((Member)other).data);
    }

    @Override
    public int hashCode() {
        int h = hashcode;
        if (h == -1) {
            h = Arrays.hashCode(data);
            hashcode = h;
        }
        return h;
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }

}
