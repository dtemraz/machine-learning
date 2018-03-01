package examples.stocks;

import supervised.learning.samples.LearningSample;
import supervised.learning.algorithms.DeltaRuleGradientDescent;
import supervised.neuron.Activation;
import supervised.neuron.Neuron;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class tests delta rule as value estimator(linear regression) for stock prices.
 * There is a file "stocks.txt" which defines price movement of a given stock over the course of ~250 days.
 * {@link LearningSample} samples are defined with input which is price in previous N - 1 days, and desired output; price in Nth day.
 *
 * <p>
 * Since our model is very simple, consisting of only one supervised.neuron, the quality of estimation will be rather poor. With this setup
 * we cannot catch dependencies between variables, this would require hidden layers.
 * Neuron will be able to represent seen samples with pretty good accuracy but estimation for unseen samples will be bad.
 * </p>
 */
public class TestStocks {

    private static final double lEARNING_RATE = 5.6908e-09;
    private static final double ERROR_TOLERANCE = 0.000001;
    private static final double MAX_EPOCH = 200_000;

    private static final int PREVIOUS = 80;
    private static final int SAMPLE_SIZE = 70;


    public static void main(String[] args) throws Exception {

        double[] prices = loadPrices();
        StockHistory history = new StockHistory(prices);

        Neuron neuron = new Neuron(PREVIOUS, Activation.IDENTITY::apply, new DeltaRuleGradientDescent(lEARNING_RATE, ERROR_TOLERANCE, MAX_EPOCH));
        neuron.train(samples(history, PREVIOUS, SAMPLE_SIZE));

        System.out.println("seen examples");
        verifySamples(neuron, history, prices,PREVIOUS + 1, PREVIOUS + SAMPLE_SIZE);

        System.out.println("unseen examples");
        verifySamples(neuron, history, prices,PREVIOUS + SAMPLE_SIZE, PREVIOUS + SAMPLE_SIZE + 11);
    }


    private static void verifySamples(Neuron neuron, StockHistory history, double[] prices, int from, int to) {
        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
        for (int price = from; price < to; price++) {
            String estimation = decimalFormat.format(neuron.output(history.previousPrices(price, PREVIOUS)));
            System.out.println("real: " + prices[price] + ", estimated: " + estimation);
        }
    }

    private static List<LearningSample> samples(StockHistory stock, int size, int samplesCount) {
        List<LearningSample> learningSamples = new ArrayList<>();
        for (int price = 0; price < samplesCount; price++) {
            double[] previous = Arrays.copyOfRange(stock.prices, price, size + price);
            learningSamples.add(new LearningSample(previous, stock.prices[size + price]));
        }
        return learningSamples;
    }

    private static double[] loadPrices() throws IOException {
        String[] priceHistory;
        try (BufferedReader reader = new BufferedReader(new FileReader(new File("src/test/resources/stocks.txt")))) {
            priceHistory = reader.readLine().split(",");
        }
        return Arrays.stream(priceHistory).mapToDouble(Double::valueOf).toArray();
    }

}
