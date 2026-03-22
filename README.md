# MarkovQuant: Market Regime Risk Engine

A quantitative framework for detecting market regimes, quantifying regime-conditional risk, and simulating forward-looking price paths. Applies Markov chain theory to equity, ETF, and cryptocurrency markets using live price data.

## Overview

Financial markets cycle through structurally distinct periods that differ in return distribution, volatility, and tail risk. Standard risk models treat all historical observations as homogeneous, which underestimates risk during adverse regimes and overstates it during calm ones. MarkovQuant models these structural shifts explicitly using a Hidden Markov Model, estimates transition probabilities between regimes via Maximum Likelihood Estimation, and conditions all downstream risk metrics on the detected regime.

The framework consists of six components:

1. HMM-based regime detection
2. Markov transition matrix estimation and analysis
3. Regime-conditional VaR, CVaR, and formal backtesting via Kupiec POF and Christoffersen Conditional Coverage
4. Walk-forward backtesting with vol-scaled transaction costs
5. Forward probability projections and Block Bootstrap Monte Carlo simulation
6. Stationarity testing via rolling re-estimation and chi-squared homogeneity

## Mathematical Foundation

### Regime Detection: Gaussian HMM

Regimes are treated as latent (unobserved) states. The model is fitted on two features: daily return and 21-day rolling realized volatility. The EM algorithm iterates between an E-step (computing posterior state probabilities given parameters) and an M-step (updating parameters to maximize likelihood) until convergence. Each state has its own diagonal Gaussian emission distribution, fitted jointly with the transition structure.

States are sorted by mean return post-fitting and labeled Bear, Neutral, and Bull. The label mapping is learned on training data only. When the model is applied to held-out data, training-set feature scaling is applied, which is the correct procedure to prevent data leakage.

### Transition Matrix: MLE

$$\hat{P}_{ij} = \frac{N_{ij}}{\sum_k N_{ik}}$$

$N_{ij}$ is the observed count of consecutive day pairs where regime $i$ was followed by regime $j$. Dividing by the row total gives transition probabilities. Each row sums to 1.

### Chapman-Kolmogorov: N-step Projections

$$P^{(n)} = P^n$$

Raising the transition matrix to the power $n$ gives the probability of being in each regime $n$ trading days from now. This is used in the forward probability fan chart and in the forward VaR calculation.

### Stationary Distribution

$$\pi P = \pi, \quad \sum_i \pi_i = 1$$

Solved via the left eigenvector corresponding to eigenvalue 1. Represents the long-run fraction of time the market spends in each regime, independent of starting state. Mean recurrence time for regime $i$ is $1/\pi_i$.

### Regime-Conditional VaR and CVaR

$$\text{VaR}_\alpha^{(r)} = \inf \{ x : P(R \leq x \mid \text{Regime} = r) \geq \alpha \}$$

$$\text{CVaR}_\alpha^{(r)} = E[R \mid R \leq \text{VaR}_\alpha^{(r)}, \text{Regime} = r]$$

VaR is estimated separately per regime from the empirical return distribution of days assigned to that regime. CVaR is the mean of returns in the tail below VaR.

### Forward Probability-Weighted VaR

$$\text{VaR}^{\text{fwd}}(h) = \sum_j P^h_{i_0, j} \cdot \text{VaR}^{(j)}$$

At horizon $h$, the current regime $i_0$ may have transitioned to any other regime. The forward VaR weights each regime's VaR estimate by the Chapman-Kolmogorov probability of being in that regime at horizon $h$.

### VaR Backtesting: Kupiec POF Test

$$LR_{pof} = -2 \left[ (T-N)\log(1-\alpha) + N\log(\alpha) - (T-N)\log(1-\hat{p}) - N\log(\hat{p}) \right] \sim \chi^2_1$$

Tests whether the observed breach rate $\hat{p} = N/T$ is statistically consistent with the claimed confidence level $\alpha$. Reject if $LR_{pof} > 3.84$. Estimated on training data, evaluated on held-out test data only.

### VaR Backtesting: Christoffersen Conditional Coverage

$$LR_{cc} = LR_{pof} + LR_{ind} \sim \chi^2_2$$

Kupiec tests frequency only. Christoffersen (1998) adds an independence test on the binary breach sequence. Let $n_{01}$ be the count of breaches following a calm day and $n_{11}$ breaches following a breach day. If $\hat{p}_{11} \gg \hat{p}_{01}$, violations are clustering, which means the model systematically underestimates risk during crises. The combined statistic $LR_{cc}$ tests both correct coverage and independent violations. Reject if $LR_{cc} > 5.99$.

### Block Bootstrap Monte Carlo

At each simulation step, a block of 10 consecutive historical returns is sampled from the current regime's empirical return pool. After each block, the next regime is drawn from the Markov transition matrix. This preserves serial dependence (volatility clustering, momentum) that independent-day sampling would destroy. Regimes with fewer than 50 historical observations fall back to a Student-t distribution with MLE-fitted degrees of freedom.

### Stationarity Testing

The entire model assumes transition probabilities are constant over time. This is tested two ways. First, the transition matrix is re-estimated quarterly on an expanding window to track probability drift visually. Second, a chi-squared homogeneity test compares transition count matrices from the first half of the sample against the second half. For each from-state a 2x3 contingency table is built and $\chi^2$ is applied. A p-value below 0.05 indicates the transition structure changed, violating the stationarity assumption.

## Design Decisions

**Three fixed states.** The model uses Bear, Neutral, and Bull throughout. Fewer states lose meaningful risk differentiation; more states reduce observations per state and make MLE estimates unreliable on typical 5-10 year datasets.

**Diagonal covariance in HMM.** The full covariance matrix requires $O(d^2)$ parameters per state and frequently becomes singular on datasets of this size. Diagonal covariance is numerically stable and approximately valid given that daily return and rolling volatility are weakly correlated within a regime.

**Feature standardization.** Daily returns are on the order of 0.001 and rolling vol on the order of 0.01. Feeding raw values to the EM algorithm produces ill-conditioned covariance matrices. Z-score normalization using training-set statistics is applied before fitting and before any prediction.

**Five EM restarts.** hmmlearn does not support multiple initializations natively. The model is fitted from five different random seeds and the highest log-likelihood solution is kept, reducing sensitivity to local optima.

**Lookahead-safe backtesting.** The HMM is re-fitted on training prices only and the frozen model is used to predict regimes on held-out test prices. If the full-sample HMM labels were used for the test period, future information would contaminate the signal.

**Vol-scaled transaction costs.** Bid-ask spreads widen during high-volatility regimes. The base cost of 5 bps is scaled by the ratio of current 21-day volatility to mean volatility, making friction larger precisely when the model is most likely to whipsaw.

**Out-of-sample Kupiec and Christoffersen.** VaR is estimated on the training period. Both tests count breaches on the held-out test period only. Testing on the estimation sample would be circular.

## Project Structure

```
app.py              main Streamlit application
requirements.txt    Python dependencies
README.md           this file
```

The application is organized into five classes:

```
MarkovEngine         MLE fitting, n-step matrix, stationary distribution, mean recurrence time
RegimeClassifier     Gaussian HMM with fit/predict separation for lookahead safety
RiskEngine           VaR, CVaR, Kupiec, Christoffersen, forward VaR
Backtester           Walk-forward backtest with vol-scaled friction
MonteCarloSimulator  Block bootstrap regime-conditional simulation
StationarityAnalyser Rolling re-estimation and chi-squared homogeneity test
```

## References

Christoffersen, P. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841-862.

Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series. Econometrica, 57(2), 357-384.

Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives, 3(2), 73-84.

UI is inspired from Roman Paolucci's SLMC Model. https://slmc-model-rmp.streamlit.app 


