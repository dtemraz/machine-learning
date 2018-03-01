package examples.iris;

import java.util.Arrays;

class Iris {
    
    Iris(double[] components, IrisType type) {
        this.components = components;
        this.type = type;
    }
    
    private final double[] components;
    private final IrisType type;
    
    double[] getComponents() {
        return components;
    }
    
    IrisType getType() {
        return type;
    }

    @Override
    public String toString() {
        return type.toString() + ": " + Arrays.toString(components);
    }
    
}
