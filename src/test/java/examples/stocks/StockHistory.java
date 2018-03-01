package examples.stocks;

import java.util.Arrays;

class StockHistory {

    final double[] prices;

    StockHistory(double[] prices) {
        this.prices = prices;
    }

    /**
     * Returns previous day prices for the given <em>day</em>, starting from
     * earliest price, up to current day(exclusive).
     *
     * @param day for which to return previous days
     * @param previousDays number of previous days for which price should be returned
     * @return previous day prices up to current day(exclusive)
     */
    double[] previousPrices(int day, int previousDays) {
        if (day == previousDays) {
            return null;
        }
        return Arrays.copyOfRange(prices, day - previousDays, day);
    }
}
