# Group Project - Advanced Programming

## Options Calculator

Goal of this project is to implement a fully-functioning, interactive Option Calculator, that for a given input combination of an Underlying Asset, a strike price and some more information, will return/display the current option price, as well as all the relevant greeks ($\delta, \gamma, \rho, \theta$, etc.).

Although the Calculator need not be in an interactive environment, it should in theory have all the capabilities of the one, one can find here: https://www.cboe.com/education/tools/options-calculator/

The option calculator should work with European, American, Asian, Binary, and Barrier options, where the option-type can be specified as an input.
For those types without a closed formula, the Black-Scholes Model should be used to calculate the options price. 
In Monte-Carlo, the number of simulations, as well as the time step should be parameters of the model. Input, and all other parameters must always be read from file. The program should by default write to the console, but it should be possible to write to a file.

### 0.0) Inputs

| **Input** | **Data Type** | **Values** |
|-------------------|-------------------|-------------------|
| Option Type       | string            |['call', 'put']    |
| Exercise Type     | string            |['american', 'european', 'asian', 'binary', 'barrier']|
| Start Date        | date (or int)     | -                 |
| Start Time        | time (or int)     | -                 |
| Expiration Date   | date (or int)     | -                 |
| Expiration Time   | string            | ['AM', 'PM']      |
| Option Strike     | float             | -                 |
| Stock/Underlying Price| float         | -                 |
| Option Volatility | float (in %)      | -                 |
| Interest Rate     | float (in %)      | -                 |
| Dividends (optional) | string         | ['dividend', 'dividend stream']|
| Dividend amount   | float (in $)      | -                 |
| Day interval (for dividend stream)| int | -               |
| Nr. of simulations (MC) | int         | -                 |
| Time-Step (MC)    | float (in days?)  | -                 |



### 0.1) Outputs

| **Output** | **Data Type** |
|-------------------|-------------------|
| Theoretical Price | float             |
| Delta ($\delta$)  | float             |
| Gamma ($\gamma$)  | float             |
| Rho   ($\rho$)    | float             |
| Theta ($\theta$)  | float             |
| Vega  ($\nu$)     | float             |



### 0.2) Option Types

What is an Option: Right to Buy or Sell an Asset at a certain time for a certain price. A Call-Option grants the owner of the option the right to **buy** the underlying asset, a Put-Option the right to **sell** the underlying asset (at certain time for certain price). 

Payoff of an Option: At expiration, an option can be either in the money (ITM), meaning that for a **Call-Option**, the price of the underlying is higher than the strike price of the option, so that the owner of the option can exercise the option and make profit: $S_T - K_0$, where $S_T$: Price of the underlying, $K_0$: Strike Price and for a **Put-Option**, the price of the underlying is lower than the strike price of the option, so that the owner of the option can exercise the option and make profit: $K_0 - S_T$.

There exist different types of options. They differ in regard to the exercising rights, the payoff, as well as other dynamics. 
|**Option Type** | **Characteristics** |
|----------------|---------------------|
| European       | Can only be exercised at Maturity |
| American       | Can be exercised at any point before Maturity |
| Asian          | Payoff is calculated based on the average price of the underlying over a certain time period |
| Binary         | Payoff is dependent on a "Yes-No" proposition. Pay either fixed amount or nothing |
| Barrier        | Payoff/Existence is dependent on the Underlying reaching a certain price level |

*Note on Asian Options*: Asian options still have a fixed Strike Price $K_0$, just their payoff is not dependent on the Spot Price of the Asset when exercising, but the average price over a specified period, i.e. $\bar{S} - K_0$ for a Call-Option.

*Note on Barrier Options*: Barrier Options can be either *Knock-In's* or *Knock-Out's*, both of which also have regular Strike Prices and Maturities (like a European Option), only with the extra feature of the *Barrier*. A Knock-Out Option starts out and behaves like a regular European Option, as long as the price of the Underlying doesn't fall under a certain level. If it does fall under that level before expiration, the option becomes worthless immediately. Therefore Barrier Options are usually cheaper than regular options, because they have the possibility of becoming worthless before expiration. A Knock-In Option is "inactive"/worthless unless a certain Barrier price is reached. Once that price is reached, the action becomes and behaves like a normal European Option.


### 0.3) Option Pricing

Like most Assets in Finance, an option's price depend on the expected payoff of the option assuming efficient markets and No-Arbitrage. This expected payoff dependes on many factors, like the Option Strike Price, the implied volatility of the underlying (higher vol. makes it more likely, the option can "jump" ITM), the time until expiration (the more time, the higher the chance the option can move ITM), and many more. 

For basic types, like European Options, there exist closed form solutions (Black-Scholes Model), but for most options, due to their complexity, the prices are determined numerically, using Monte-Carlo Simulation or Binomial Trees.

Nevertheless, for complex option simulations, the assumptions underlying the simulation are still based on Black-Scholes pricing and Geometric Brownian Motion. 

For the purpose of this project, we will focus on using Monte-Carlo to determine the option's price. 

Generally, the process works the following: 
- Given M: number of simulations, N: number of timesteps, simulate M different price paths, the underlying asset can have until maturity. Discretize the timeframe using N. The assumption underlying this random simulation is Geometric Brownian Motion (GBM). This means that each time-step, the asset has a tiny (random) change in price, up or down. These increments are **independent** and **normally distributed**. Over time, these increments accumulate into a **continuos but unpredictable** path. 
- Each Timestep, calculate: $S_{i + \Delta t} = S_i \cdot \exp[(r - q - \frac{1}{2} \sigma^2) \Delta t + \sigma \sqrt{\Delta t} Z]$, where $Z \sim (0,1)$ is a standard normal random variable, and q is the Dividend yield. 
- For each simulated path, $m$, calculate the option's payoff at expiration: $p_m = \max(\pm S_T - K_0, 0)$
- For a European Option, the option price is the Present Value (PV) of the average of all payoffs $p_m$: $V_0 = e^{-rT} \frac{1}{M} \sum_{m=1}^{M} p_m$

- For all other options, this gets more complex, because there is the possibility of *Early Exercise*, i.e. exercising the option before expiration, if we already are ITM. This means, that at every timestep $n$ where the option is ITM, we have to account for the possibility of early exercise. For this, we have to compare the payoff if executing the option immediately (Immediate Exercise Value, IEV) with the expected payoff of holding onto it for longer (Continuation Value), where the CV is calculated as the expected discounted future cashflow. If IEV > CV, the option gets exercised, the payoff $p_m$ of the path is set as: $S_T - K_0$, and all future cashflows of this path are set to 0. Otherwise, the option is held on for longer, and the same process repeats every timestep, until it is either exercised or expired.


### 0.4) Greeks

Generally, the Greeks represent the rates of change in the option price with respect to an input variable in the pricing model:
| **Greek** | **What it measures** |
|-----------|----------------------|
|Delta ($\delta$) | sensitivity to the price of the underlying |
|Gamma ($\gamma$) | curvature (rate of change of delta) |
|Rho ($\rho$) | sensitivity to interest rates | 
| Theta ($\theta$)| time decay |
| Vega ($\nu$) | sensitivity to changes in (implied) volatility |



### *0.5) Notes*

- For Black-Scholes, time until expiration is given in years (i.e. days/365). Opinions differ whether to use Calendar Days (i.e. days/365), or Trading Days (i.e. days/252), since options can only be traded on the latter. A common approach is to annualize each metric based on what makes the most sense logically. I.e. interest rates, which accumulate each calendar day, take the 365-day year, while volatility takes the 252-day year, since assets cannot move (i.e. be volatile) on non-trading days. A common approach is: $T = \frac{days}{365}, \quad \sigma_{d} = \frac{\sigma_a}{252}, \quad r_d = \frac{r_a}{365}$
- TBD


