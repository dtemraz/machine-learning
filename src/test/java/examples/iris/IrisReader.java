package examples.iris;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

class IrisReader {

    static final int IRIS_MEASURES = 4;
    private static final int IRIS_LABEL = 4;

    private static final String IRIS_DELIMITER = ",";
        
    static Map<IrisType, List<Iris>> read(File file) {
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            List<Iris> irisCollection = new ArrayList<>();
            String line = "";
            while ((line = reader.readLine()) != null) {
                double[] components = new double[IRIS_MEASURES];
                String[] irisDetails = line.split(IRIS_DELIMITER);
                for (int i = 0; i < IRIS_MEASURES; i++) {
                    components[i] = Double.valueOf(irisDetails[i]);
                }
                
                irisCollection.add(new Iris(components, IrisType.forLabel(irisDetails[IRIS_LABEL])));
            }
            return groupByType(irisCollection);
        } 
        catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }
    
    private static Map<IrisType, List<Iris>> groupByType(List<Iris> irisCollection) {
        return irisCollection.stream().collect(Collectors.groupingBy(Iris::getType));
    }
    
}
