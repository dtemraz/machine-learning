package utilities.concurrency;

import java.lang.annotation.*;

/**
 * This is a marker annotation that denotes a class being a thread safe without any additional client code synchronization.
 *
 * @author dtemraz
 */
@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.CLASS)
public @interface ThreadSafe {

}
