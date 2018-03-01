package examples.iris;

enum IrisType {
    
    SETOSA("Iris-setosa"),
    VERSICOLOR("Iris-versicolor"),
    VIRGINICA("Iris-virginica");

    private final String label;

    private IrisType(String label) {
        this.label = label;
    }

    String getLabel() {
        return label;
    }

    static IrisType forLabel(String label) {
        if (SETOSA.label.equals(label)) {
            return SETOSA;
        }
        if (VERSICOLOR.label.equals(label)) {
            return VERSICOLOR;
        }
        if (VIRGINICA.label.equals(label)) {
            return VIRGINICA;
        }
        throw new IllegalArgumentException("no mapping for label: " + label);
    }
}