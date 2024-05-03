#  2024 IMC Global Trading Challenge
**Final Rank: 332 / 9,140 Teams**

<img src="Final Results.png" alt="Final Results" width="360"/>

\
Over 15 days, I participated in Prosperity 2, an algorithmic trading challenge hosted by IMC Trading. Participants had 72 hours for each of the  5 rounds to code a trading strategy for each round. Each round presented a distinct set of products, necessitating different trading strategies.

- [Round 1](#round-1) - Market Making
- [Round 2](#round-2) - Exchange Arbitrage
- [Round 3](#round-3) - ETF Arbitrage + Pairs Trading
- [Round 4](#round-4) - Options Trading
- [Round 5](#round-5) - Counterparty Trading 

## Round 1 - **[Submission Code](https://github.com/edmund870/2024-IMC-Global-Trading-Challenge/blob/main/Round%201/Round%201%20Submission.py)**, **[Results](https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/edmund870/2024-IMC-Global-Trading-Challenge/main/Round%201/Round%201%20Results.log)**

**Products Traded: `STARFRUIT`, `AMETHYSTS`**

In Round 1, the product `AMETHYSTS` exhibited a stable price, hovering around a fair value of 10,000. Therefore, the strategy centered on market taking and market making around this fair price.

As for `STARFRUIT`, it was noted that whenever the bid/ask price spiked beyond the previous mid-price, it was ideal to either hit the bid or lift the ask. To enhance execution, a filter condition was implemented, ensuring trades were executed only when the bid-ask spread remained below 3. In instances where none of these conditions were met, market making around the best bid and best ask was performed, undercutting both prices by 1.

The manual challenge was relatively straightforward. given a probability distribution representing the acceptable price of an undisclosed number of counterparties, the objective was to maximize the expected value by offering two price points for a potential trade. Monte Carlo simulation was performed to determine the optimal price points that would yield maximum profit.

**Round 1 Results: 581 / 9,140** 

<img src="Round 1/Round 1 Results.png" alt="Round 1 Results" width="600"/>

### Post-Round Thoughts
Round 1 results were satisfactory, there were minimal improvements that could have been done. Linear regression was suggested by the community to aid in trading. 

## Round 2  - **[Submission Code](https://github.com/edmund870/2024-IMC-Global-Trading-Challenge/blob/main/Round%202/Round%202%20Submission.py)**, **[Results](https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/edmund870/2024-IMC-Global-Trading-Challenge/main/Round%202/Round%202%20Results.log)**

**New Product Traded: `ORCHIDS`**

In round 2, `ORCHIDS` were introduced. The strategy involved identifying arbitrage between two marketplaces: Archipelago and South. Factors for consideration included import/export tariffs, transportation costs, and storage costs. 

When purchasing `ORCHIDS` from the South, trades are executed at the ask, incurring transportation fees while receiving import tariffs. 

`south_buy_price = conv_ask + transport_fees + import_tariff`

Conversely, when selling `ORCHIDS` to the South, trades are executed at the bid, incurring transportation fees and export tariffs.

`south_sell_price = conv_bid - transport_fees - export_tariff`

Analysis revealed that going short at the Archipelago and immediately going flat by buying from the South was the most profitable. This was due to the receipt of the import tariffs, leading to profits. Going long is suboptimal due to the substantial export tariffs and the associated storage costs for holding the inventory. 

Therefore, the strategy was to aggressively market make as close to the mid price of `ORCHIDS` at the Archipelago as possible to establish a short position before immediately importing `ORCHIDS` from the South to go flat in the subsequent timestamp. 

A slight adjustment to Round 1's strategy was made, incorporating linear regression to predict the next mid price of `STARFRUIT`.

For the manual challenge, the concept tested was the use of FX arbitrage to maximize profits. With FX rates provided for different products, the goal was to execute trades in the optimal sequence, starting from one currency and ending with the same currency. This was done using Permutations and Combinations (PnC) to determine the optimal trade order.

**Round 2 Results: 248 / 9,140** 

<img src="Round 2/Round 2 Results.png" alt="Round 2 Results" width="600"/>

### Post-Round Thoughts
Round 2 results boosted rankings. Satisfactory results. It was said that humidity and sunlight affected `ORCHIDS`, however, I was unable to determine how the product was affected. 

## Round 3 - **[Submission Code](https://github.com/edmund870/2024-IMC-Global-Trading-Challenge/blob/main/Round%203/Round%203%20Submission.py)**, **[Results](https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/edmund870/2024-IMC-Global-Trading-Challenge/main/Round%203/Round%203%20Results.log)**
**New Products Traded: `GIFT_BASKET`, `CHOCOLATE`, `STRAWBERRIES`, `ROSES`**

Round 3 introduced 4 products, testing the concept of ETF arbitrage and pairs trading. The ETF was `GIFT_BASKET` which contained 4 `CHOCOLATE`, 6 `STRAWBERRIES`, and 1 `ROSES`. 

Analysis showed that `GIFT_BASKET` typically traded at a mean premium of 380 with a standard deviation premium of 76 from its underlying components. Therefore, the idea behind the strategy was to long `GIFT_BASKET` when it trades below the premium (undervalued) and short `GIFT_BASKET` when it trades above the premium (overvalued). To further refine the trade execution, the conversion of the premium to z-score was used where
`z = (premium - 380) / 76` and the trade was executed only if the z-score exceeded or fell below the 75th percentile.

Pairs trading was also used to trade `ROSES` and `CHOCOLATE`, as well as `CHOCOLATE` and `STRAWBERRIES`. It was observed that these product pairs tended to exhibit opposing price movements. The bet on mean reversion was the strategy here. Similar z-scoring methodologies were applied in pairs trading analysis.

Round 3's manual challenge depended on trader psychology and modelling was not feasible. Profitability depended on a sharing mechanism where if more traders picked the same spot, profits would be shared among a bigger pool of participants, resulting in lower profits.

**Round 3 Results: 266 / 9,140**

<img src="Round 3/Round 3 Results.png" alt="Round 3 Results" width="600"/>

### Post-Round Thoughts
Both strategies failed to yield positive returns, resulting in negative profits for the round. Upon reflection, refining the entry and exit strategy for `GIFT_BASKET` could have improved the results. Hardcoding the mean and standard deviation of the premium was not a good idea as post-result analysis revealed that the mean premium fell from 380 to 340. Additionally, abstaining from trading the underlying products might have been a better decision.

## Round 4 - **[Submission Code](https://github.com/edmund870/2024-IMC-Global-Trading-Challenge/blob/main/Round%204/Round%204%20Submission.py)**, **[Results](https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/edmund870/2024-IMC-Global-Trading-Challenge/main/Round%204/Round%204%20Results.log)**
**New Products Traded: `COCONUT`, `COCONUT_COUPON`**

Round 4 tested the concept of option pricing. `COCONUT_COUPON` were $10,000 call options on `COCONUT` expiring 250 days. The first thing that came to mind is the Black-Scholes Option Pricing Formula. 

With the initial `COCONUT_COUPON` price on the first timestamp set as 637.63, Implied Volatility (IV) was derived to be approximately 19.33% using the following parameters: `risk-free rate: 0%`, `time to maturity: 250 / 365`, `strike: 10,000`, `initial price: 10,000`.

The trading strategy for this round entailed purchasing `COCONUT_COUPON` if the theoretical call price is above the current price, and conversely, selling `COCONUT_COUPON` if the theoretical call price is below the current price. A threshold was also imposed, triggering trades only if the percentage difference between the theoretical call price and the current call price exceeded 1.8%.

Although the IV was computed at T<sub>0</sub>, to avoid hardcoding parameters and a repeat of Round 3, IV was computed using Newton-Raphson method and cached for each iteration. The mean IV across a time period of 200 iterations was used to compute the call price.

Delta hedging was also employed to hedge against unfavorable movements in the option price. Delta of `COCONUT_COUPON` was computed and positions in `COCONUT` were adjusted accordingly: long if `COCONUT_COUPON` was short, and vice versa.

Refinements were also made to Round 3 strategy, incorporating rolling means of the premium / spread for the respective products. 

It was noticed that the function for humidity is a smoothed graph and `ORCHIDS` prices moved in the same direction as humidity. A condition, identifying the max/min point of humidity was introduced to determine the trade of `ORCHIDS`.

**Round 4 Results: 280 / 9,140** 

<img src="Round 4/Round 4 Results.png" alt="Round 4 Results" width="600"/>

### Post-Round Thoughts
Result from rRound 4 products was commendable but could have been better on 2 adjustments.
1. The condition to trade the option was too tight at a 1.8% difference, resulting in little trade execution
2. Delta hedging was a conservative approach that limited profits

Round 3 products were barely profitable even with the removal of the hardcoded parameters.

## Round 5 - **[Submission Code](https://github.com/edmund870/2024-IMC-Global-Trading-Challenge/blob/main/Round%205/Round%205%20Submission.py)**, **[Results](https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/edmund870/2024-IMC-Global-Trading-Challenge/main/Round%205/Round%205%20Results.log)**
**New Product Traded: -**

**Counterparty Trades were revealed**

In Round 5, counterparty trades were revealed, allowing for the improvement of strategies.

`Rhianna` was the standout observation who constantly capitalized on favorable market movements by buying low and selling high with `ROSES`. Similar behavior was also observed when she trades `STARFRUIT` with quantities >5. Hence, trade copying was implemented to mimick her trades.

Apart from `Rhianna`, the majority of the other counterparties' trades appeared to be noise, with no noticable patterns.

Adjustments for Round 3: Relative Strength Index (RSI) with varying window length was implemented for `CHOCOLATE` and `STRAWBERRIES` as it demonstrated promising results during backtesting.

Adjustments for Round 4: The trade execution threshold for `COCONUT_COUPON` was relaxed, from 1.8% to 0.1%. Delta hedging using `COCONUT` was discontinued. Instead, `COCONUT` was used to make leveraged bets. The rationale is that when `COCONUT_COUPON` is undervalued, price is expected to increase. Since prices of the option are dependent on the underlying, prices of `COCONUT` are also expected to increase. Therefore, directional bets were placed on both `COCONUT_COUPON` and `COCONUT`.


**Round 5 Results: 332 / 9,140** 

<img src="Round 5/Round 5 Results.png" alt="Round 5 Results" width="600"/>

### Post-Round Thoughts

Copying `Rhianna` proved successful, with `ROSES` yielding profits of 22k. Implementing RSI strategy on `CHOCOLATE` and `STRAWBERRY` resulted in modest gains, consistent with backtesting. Unfortunately, `GIFT_BASKET` weighed down overall profitability, incurring losses of 60k. Although `COCONUT_COUPON` remained profitable, it appeared that hardcoding IV allowed others to capitalize on higher profits. Additionally, the leveraged bet on `COCONUT` did not do well, resulting in losses of 17k.

In hindsight, it may have been wiser to abstain from trading `GIFT_BASKET` due to the inability to devise a profitable strategy, and to exercise restraint with `COCONUT` to prevent excessive greed.

### Final Thoughts

I would like to extend my gratitude to IMC Trading for organizing this trading challenge. It provided me with a glimpse into the world of algorithmic trading, where we design and implement trading strategies while navigating market uncertainties. It has been an invaluable and memorable experience.

Hoping for a Prosperity 3 next year!
