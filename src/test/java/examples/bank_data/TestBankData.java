package examples.bank_data;

import algorithms.cart.ClassificationTree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author dtemraz
 */
public class TestBankData {

    public static void main(String[] args) {
        List<String[]> bankData = loadData();

        List<double[]> data = bankData.stream().map(TestBankData::bla).collect(Collectors.toList());

        Map<Double, List<double[]>> categories = data.stream().collect(Collectors.groupingBy(row -> row[row.length - 1]));

        int size0 = categories.get(0d).size();
        int size1 = categories.get(1d).size();

        int totalSamples = size0 + size1;
        double dif = size1 / (double) (size0);
        int validationSamples = (int) (totalSamples * 0.25);
        int validation0 = (int) (validationSamples / (1 + 1 * dif));
        int validation1 = validationSamples - validation0;

        System.out.println();


        List<double[]> validationZero = new ArrayList<>();

        Iterator<double[]> it = categories.get(0d).iterator();
        int step = 0;
        while (it.hasNext()) {
            validationZero.add(it.next());
            it.remove();
            if (++step == validation0) {
                break;
            }
        }

        List<double[]> validationOne = new ArrayList<>();

        it = categories.get(1d).iterator();
        step = 0;
        while (it.hasNext()) {
            validationOne.add(it.next());
            it.remove();
            if (++step == validation1) {
                break;
            }
        }

        List<double[]> completeValidation = new ArrayList<>(validationZero);
        completeValidation.addAll(validationOne);

        data = categories.values().stream().flatMap(list -> list.stream()).collect(Collectors.toList());

        ClassificationTree tree = new ClassificationTree(data, 10, 20);
        int correct = 0;
        for (double[] row : completeValidation) {
            double expected = row[row.length - 1];
            double predicted = tree.classify(row);
            System.out.println("expected: " + expected + " , predicted: " + predicted);
            if (expected == predicted) {
                correct++;
            }
        }

        System.out.println("overall accuracy: " + (correct / (double) completeValidation.size()));

        System.out.println();
    }


    private static List<String[]> loadData() {
        List<String[]> bankData = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader("src/test/resources/data_banknote_authentication.txt"))) {
            String line = "";
            while ((line = reader.readLine()) != null) {
                bankData.add(line.split(","));
            }
        } catch (IOException e) {
            throw new IllegalStateException("error reading sms file ", e);
        }
        return bankData;
    }

    private static double[] bla(String[] bankNote) {
        double[] d = new double[bankNote.length];
        for (int i = 0; i < bankNote.length; i++) {
            d[i] = Double.parseDouble(bankNote[i]);
        }
        return d;
    }

}
