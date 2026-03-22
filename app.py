"""
MarkovQuant: Market Regime Risk Engine

Core methodology used in the project: 
  - Regime detection : Gaussian Hidden Markov Model 
  - Monte Carlo      : Historical Block Bootstrap (non-parametric, preserves autocorrelation)
                       with Student-t fallback for regimes with little data 
  - Backtesting      : Volatility-scaled transaction costs 
  - VaR validation   : Kupiec POF Test 
  - Stationarity     : Rolling transition matrix + chi-squared homogeneity test
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

class MarkovEngine:
    """
    MLE estimation of Markov transition matrix from a sequence of states.
    """

    def __init__(self, states: list):
        self.states = states
        self.transition_matrix: pd.DataFrame = None
        self.transition_counts: pd.DataFrame = None

    def fit(self, state_sequence: pd.Series) -> pd.DataFrame:
        """
        MLE: P_ij = N_ij / sum_k N_ik
        Walking through every consecutive pair of states to count and store transitions. 
        Dividing each row by its total to get probabilities.
        """
        counts = pd.DataFrame(0, index=self.states, columns=self.states, dtype=float)
        seq = state_sequence.dropna()
        for t in range(len(seq) - 1):   # n-1 transitions from n states
            s0, s1 = seq.iloc[t], seq.iloc[t + 1]
            if s0 in self.states and s1 in self.states:
                counts.loc[s0, s1] += 1
        self.transition_counts = counts.copy()
        row_sums = counts.sum(axis=1)
        self.transition_matrix = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
        return self.transition_matrix

    def n_step_matrix(self, n: int) -> pd.DataFrame:
        """Chapman-Kolmogorov: P^(n) = P^n"""
        P_n = np.linalg.matrix_power(self.transition_matrix.values, n)
        return pd.DataFrame(P_n, index=self.states, columns=self.states)

    def stationary_distribution(self) -> pd.Series:
        """Solving pi P = pi via left eigenvector for eigenvalue 1."""
        vals, vecs = np.linalg.eig(self.transition_matrix.values.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pd.Series(pi, index=self.states)

    def mean_recurrence_time(self) -> pd.Series:
        """Expected steps to return to state i = 1 / pi_i"""
        pi = self.stationary_distribution()
        return (1.0 / pi.replace(0, np.nan)).rename("Mean Recurrence Time (days)")

    def absorption_probability(self, absorbing_state: str) -> pd.Series:
        """
        Probability of ever reaching absorbing_state from each starting state.
        """
        non_abs = [s for s in self.states if s != absorbing_state]
        Q = self.transition_matrix.loc[non_abs, non_abs].values
        r = self.transition_matrix.loc[non_abs, absorbing_state].values
        try:
            b = np.linalg.solve(np.eye(len(non_abs)) - Q, r)
        except np.linalg.LinAlgError:
            b = np.zeros(len(non_abs))
        result = pd.Series(index=self.states, dtype=float)
        result[absorbing_state] = 1.0
        for i, s in enumerate(non_abs):
            result[s] = np.clip(b[i], 0, 1)
        return result

class RegimeClassifier:
    """
    Gaussian Hidden Markov Model regime detection.
    """

    N_STATES = 3

    def __init__(self):
        self._model        = None
        self._name_map     = None
        self._feature_mean = None
        self._feature_std  = None

    def fit(self, prices: pd.Series) -> "RegimeClassifier":
        if not HMM_AVAILABLE:
            st.error("hmmlearn not installed. Run `pip install hmmlearn`.")
            st.stop()

        returns  = prices.pct_change().dropna()
        X_raw    = self._features(returns)

        # Z-score standardise: mean=0, std=1 per feature
        self._feature_mean = X_raw.mean(axis=0)
        self._feature_std  = X_raw.std(axis=0).clip(min=1e-8)
        X = (X_raw - self._feature_mean) / self._feature_std

        # Run EM from multiple random seeds, keeping best result
        best_model  = None
        best_score  = -np.inf
        for seed in [42, 7, 13, 99, 21]:
            try:
                m = GaussianHMM(
                    n_components=self.N_STATES,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=seed,
                )
                m.fit(X)
                score = m.score(X)
                if score > best_score:
                    best_score = score
                    best_model = m
            except Exception:
                continue

        if best_model is None:
            st.error("HMM fitting failed across all initialisations. Try extending the date range (5+ years recommended).")
            st.stop()

        model = best_model

        # Determine label mapping on training data only
        hidden = model.predict(X)
        state_means = {s: returns.values[hidden == s].mean() for s in range(self.N_STATES)}
        sorted_states = sorted(state_means, key=state_means.get)
        self._name_map = {
            sorted_states[0]: "Bear",
            sorted_states[1]: "Neutral",
            sorted_states[2]: "Bull",
        }
        self._model = model
        return self

    def predict(self, prices: pd.Series) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        returns = prices.pct_change().dropna()
        X_raw   = self._features(returns)
        X       = (X_raw - self._feature_mean) / self._feature_std
        hidden  = self._model.predict(X)
        return pd.Series([self._name_map[s] for s in hidden], index=returns.index)

    def classify(self, prices: pd.Series) -> pd.Series:
        return self.fit(prices).predict(prices)

    @staticmethod
    def _features(returns: pd.Series) -> np.ndarray:
        """
        Feature matrix for HMM: [daily return, 21-day rolling vol].
        21-day vol is the institutional standard which captures structural
        volatility regime shifts rather than noise in very short windows.
        """
        rv = returns.rolling(21).std().fillna(returns.std())
        return np.column_stack([returns.values, rv.values])

class RiskEngine:
    """
    Regime-conditional VaR, CVaR, Kupiec test, and forward VaR.
    """

    def __init__(self, returns: pd.Series, regimes: pd.Series):
        self.returns = returns
        self.regimes = regimes
        self.aligned = pd.DataFrame({"r": returns, "regime": regimes}).dropna()

    def _data(self, regime: str = None) -> pd.Series:
        if regime:
            return self.aligned[self.aligned["regime"] == regime]["r"].dropna()
        return self.aligned["r"].dropna()

    def var(self, alpha: float = 0.05, regime: str = None) -> float:
        return float(np.percentile(self._data(regime), alpha * 100))

    def cvar(self, alpha: float = 0.05, regime: str = None) -> float:
        v = self.var(alpha, regime)
        tail = self._data(regime)
        tail = tail[tail <= v]
        return float(tail.mean()) if len(tail) else v

    def regime_summary(self) -> pd.DataFrame:
        rows = []
        ann = np.sqrt(252)
        for reg in self.aligned["regime"].unique():
            sub = self._data(reg)
            rows.append({
                "Regime": reg,
                "Observations": len(sub),
                "Mean Ret (Ann %)": round(sub.mean() * 252 * 100, 2),
                "Volatility (Ann %)": round(sub.std() * ann * 100, 2),
                "Sharpe (Ann)": round((sub.mean() / sub.std()) * ann, 2) if sub.std() > 0 else 0,
                "VaR 95% (%)": round(self.var(0.05, reg) * 100, 3),
                "CVaR 95% (%)": round(self.cvar(0.05, reg) * 100, 3),
                "Max Daily Loss (%)": round(sub.min() * 100, 2),
                "Max Daily Gain (%)": round(sub.max() * 100, 2),
            })
        return pd.DataFrame(rows).set_index("Regime")

    @staticmethod
    def christoffersen_test(hits: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Christoffersen (1998) Conditional Coverage Test.

        Kupiec only checks breach frequency. It misses clustered violations —
        10 consecutive breach days followed by calm would pass Kupiec (right
        total count) but clearly shows the model fails to warn before crises.

        This test adds an independence component on top of Kupiec:
          - Build transition counts of the binary breach sequence I_t
          - Test H0: p_01 == p_11 (breach prob doesn't depend on yesterday)
          - LR_ind ~ chi-sq(df=1), critical value 3.84

        Combined conditional coverage:
          LR_cc = LR_pof + LR_ind ~ chi-sq(df=2), critical value 5.99

        A model passing CC has both correct frequency AND independent breaches.
        This looks much more like a reliable VaR model that won't miss clustered crises.
        """
        T = len(hits)
        if T < 30:
            return {"error": f"Too few observations ({T}) — need >= 30"}

        N     = int(hits.sum())
        p_hat = N / T

        if p_hat in (0.0, 1.0):
            return {"error": "Edge case — 0 or all breaches, CC undefined"}

        # Kupiec POF in log-space (numerically stable)
        LR_pof = -2 * (
            (T - N) * np.log(1 - alpha) + N * np.log(alpha) -
            (T - N) * np.log(1 - p_hat) - N * np.log(p_hat)
        )

        # Transition counts of the hit sequence
        n_00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
        n_01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
        n_10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
        n_11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

        p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
        p_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0

        if p_01 in (0.0, 1.0) or p_11 in (0.0, 1.0):
            LR_ind = 0.0
        else:
            L_null = (
                (n_00 + n_10) * np.log(1 - p_hat) +
                (n_01 + n_11) * np.log(p_hat)
            )
            L_alt = (
                n_00 * np.log(1 - p_01) + n_01 * np.log(p_01) +
                n_10 * np.log(1 - p_11) + n_11 * np.log(p_11)
            )
            LR_ind = -2 * (L_null - L_alt)

        LR_cc  = LR_pof + LR_ind
        pv_pof = 1 - stats.chi2.cdf(LR_pof, df=1)
        pv_ind = 1 - stats.chi2.cdf(LR_ind, df=1)
        pv_cc  = 1 - stats.chi2.cdf(LR_cc,  df=2)

        return {
            "T": T, "N": N, "p_hat": round(p_hat, 4),
            "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11,
            "p_01": round(p_01, 4), "p_11": round(p_11, 4),
            "LR_pof": round(LR_pof, 4),
            "LR_ind": round(LR_ind, 4),
            "LR_cc":  round(LR_cc,  4),
            "pv_pof": round(pv_pof, 4),
            "pv_ind": round(pv_ind, 4),
            "pv_cc":  round(pv_cc,  4),
            "reject_pof": LR_pof > 3.84,
            "reject_ind": LR_ind > 3.84,
            "reject_cc":  LR_cc  > 5.99,
            "verdict_pof": "❌ Rejected" if LR_pof > 3.84 else "✅ Not Rejected",
            "verdict_ind": "❌ Rejected" if LR_ind > 3.84 else "✅ Not Rejected",
            "verdict_cc":  "❌ Rejected" if LR_cc  > 5.99 else "✅ Not Rejected",
        }

    def kupiec_test(self, alpha: float = 0.05, regime: str = None) -> dict:
        """
        Kupiec Proportion of Failures (POF) Test.

        Formally tests whether the observed VaR breach rate is statistically
        consistent with the claimed confidence level.

        H0: p_hat == alpha implies that model is correctly calibrated
        Reject H0 if LR > 3.84  (chi-squared critical value, df=1)

        LR = -2 * ln[ (1-p)^(T-N) * p^N  /  (1-p_hat)^(T-N) * p_hat^N ]
        """
        data = self._data(regime)
        T = len(data)
        if T < 30:
            return {"error": f"Too few observations ({T}) — need ≥ 30"}

        var_est = self.var(alpha, regime)
        N = int((data < var_est).sum())
        p_hat = N / T

        if p_hat in (0.0, 1.0):
            return {
                "T": T, "N": N, "expected_N": round(T * alpha, 1),
                "p_hat": round(p_hat, 4), "LR": None, "p_value": None,
                "rejected": False,
                "verdict": "⚠️ Edge case (0 or all breaches) — LR undefined",
            }

        LR = -2 * (
            np.log(((1 - alpha) ** (T - N)) * (alpha ** N)) -
            np.log(((1 - p_hat) ** (T - N)) * (p_hat ** N))
        )
        p_value = 1 - stats.chi2.cdf(LR, df=1)
        rejected = LR > 3.84

        return {
            "T": T, "N": N,
            "expected_N": round(T * alpha, 1),
            "p_hat": round(p_hat, 4),
            "LR": round(LR, 4),
            "p_value": round(p_value, 4),
            "rejected": rejected,
            "verdict": (
                "❌ REJECTED — VaR model miscalibrated" if rejected
                else "✅ NOT REJECTED — VaR model well calibrated"
            ),
        }

    def forward_var(self, current_regime: str, horizon: int,
                    transition_matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        Probability-weighted forward VaR/CVaR using Chapman-Kolmogorov.

        In h trading days, we can  be in any regime with probability P^h.
        Forward VaR = sum over regimes of (prob of being in regime j) * VaR_j.
        This gives a risk estimate that accounts for possible regime transitions.
        """
        states = list(transition_matrix.index)
        P_n = np.linalg.matrix_power(transition_matrix.values, horizon)
        cur_idx = states.index(current_regime)
        probs = P_n[cur_idx]
        w_var  = sum(probs[i] * self.var(alpha, s)  for i, s in enumerate(states))
        w_cvar = sum(probs[i] * self.cvar(alpha, s) for i, s in enumerate(states))
        return {
            "weighted_var": w_var,
            "weighted_cvar": w_cvar,
            "regime_probs": dict(zip(states, probs)),
        }

class Backtester:
    """
    Walk-forward regime momentum strategy with vol-scaled transaction costs.

    Why friction matters:
      Regime-switching models are prone to "whipsawing" or rapidly flipping
      between states. Each flip costs money (bid-ask spread, market impact).
      Ignoring this makes the strategy look far better than it is in reality.

    Transaction cost model:
      Instead of a fixed cost, we scale by realized volatility because
      bid-ask spreads widen in high-vol regimes (market makers charge more
      for bearing risk). Cost = base_spread * (current_vol / mean_vol).

    Lookahead guard:
      Signal is lagged by 1 day. Using today's regime to trade today's
      close is lookahead bias
    """

    BASE_SPREAD = 0.0005   # 5 bps base transaction cost

    def __init__(self, prices: pd.Series, regimes: pd.Series, train_pct: float = 0.70):
        self.prices    = prices
        self.regimes   = regimes
        self.train_pct = train_pct

    def run(self) -> tuple:
        rets = self.prices.pct_change().dropna()
        df   = pd.DataFrame({"r": rets, "regime": self.regimes}).dropna()

        split       = int(len(df) * self.train_pct)
        train_idx   = df.index[:split]
        test_idx    = df.index[split:]
        train, test = df.loc[train_idx], df.loc[test_idx]

        train_prices = self.prices.loc[self.prices.index <= train_idx[-1]]
        test_prices  = self.prices.loc[self.prices.index > train_idx[-1]]

        oos_classifier = RegimeClassifier()
        oos_classifier.fit(train_prices)
        oos_regimes = oos_classifier.predict(test_prices).dropna()

        # Align OOS regimes with test returns
        test = test.copy()
        test["regime"] = oos_regimes.reindex(test.index)
        test = test.dropna(subset=["regime"])

        # Fitting Markov engine on training regimes
        states = sorted(df["regime"].unique().tolist())
        eng    = MarkovEngine(states)
        eng.fit(train["regime"])

        def make_signal(regime: str) -> float:
            if "Bull" in str(regime): return  1.0
            if "Bear" in str(regime): return -1.0
            return 0.0

        test["signal"] = test["regime"].apply(make_signal).shift(1)  # lag 1

        # Volatility-scaled transaction cost
        roll_vol  = test["r"].rolling(21).std().fillna(test["r"].std())
        mean_vol  = roll_vol.mean()
        vol_scale = roll_vol / mean_vol
        trade     = test["signal"].diff().abs()
        cost      = trade * self.BASE_SPREAD * vol_scale

        test["strategy"]       = test["signal"] * test["r"] - cost
        test["strategy_gross"] = test["signal"] * test["r"]
        test["buy_hold"]       = test["r"]

        return test.dropna(), eng.transition_matrix

    @staticmethod
    def metrics(returns: pd.Series, label: str = "") -> dict:
        cum     = (1 + returns).cumprod()
        ann     = np.sqrt(252)
        sharpe  = (returns.mean() / returns.std() * ann) if returns.std() > 0 else 0
        dd      = (cum - cum.cummax()) / cum.cummax()
        calmar  = (returns.mean() * 252 / abs(dd.min())) if dd.min() != 0 else 0
        return {
            "Total Return":  f"{cum.iloc[-1] - 1:.2%}",
            "Ann. Sharpe":   f"{sharpe:.2f}",
            "Max Drawdown":  f"{dd.min():.2%}",
            "Ann. Vol":      f"{returns.std() * ann:.2%}",
            "Win Rate":      f"{(returns > 0).mean():.2%}",
            "Calmar Ratio":  f"{calmar:.2f}",
        }

class MonteCarloSimulator:
    """
    Regime-conditional Monte Carlo via Historical Block Bootstrapping.

    I chose block bootstrap over Gaussian / Student-t because:
      - Gaussian: drastically underestimates tail risk (ignores fat tails,
        volatility clustering, momentum).
      - Student-t: better tails, but still parametric — assumes a specific
        distributional shape.
      - Block bootstrap: non-parametric. Randomly samples BLOCKS of
        consecutive historical returns, preserving serial dependence
        (vol clustering, momentum) that individual-day sampling would break.

    Fallback to Student-t:
      Regimes with fewer than MIN_OBS observations don't have enough
      history to bootstrap reliably. Student-t is used there, with degrees
      of freedom fitted from the regime's empirical data via MLE.

    Regime transitions:
      After each block, the next regime is drawn from the Markov
      transition matrix, thus retaining the realistic possibility of switching regimes. 
    """

    BLOCK_SIZE = 10   # 10-day blocks preserve ~2 weeks of serial dependence
    MIN_OBS    = 50   # minimum observations to use bootstrap

    def __init__(self, returns: pd.Series, regimes: pd.Series,
                 transition_matrix: pd.DataFrame):
        self.P      = transition_matrix
        self.states = list(transition_matrix.index)
        df = pd.DataFrame({"r": returns, "regime": regimes}).dropna()

        self.regime_returns: dict = {}   # raw return arrays for bootstrap
        self.regime_t_params: dict = {}  # (df, loc, scale) for t fallback

        for s in self.states:
            sub = df[df["regime"] == s]["r"].values
            self.regime_returns[s] = sub
            if len(sub) >= 10:
                # Fit Student-t via MLE for fallback
                df_t, loc, scale = stats.t.fit(sub)
                self.regime_t_params[s] = (df_t, loc, scale)
            else:
                # Last resort: use global distribution
                df_t, loc, scale = stats.t.fit(df["r"].values)
                self.regime_t_params[s] = (df_t, loc, scale)

    def _sample_regime(self, state: str, n: int, rng) -> np.ndarray:
        """
        Sample n returns for a given regime.
        Uses block bootstrap if enough observations, else Student-t.
        """
        data = self.regime_returns[state]
        if len(data) >= self.MIN_OBS:
            # Block bootstrap: sample random starting points, take blocks
            samples = []
            while len(samples) < n:
                start = rng.integers(0, max(1, len(data) - self.BLOCK_SIZE))
                block = data[start: start + self.BLOCK_SIZE]
                samples.extend(block.tolist())
            return np.array(samples[:n])
        else:
            # Student-t fallback with MLE-fitted degrees of freedom
            df_t, loc, scale = self.regime_t_params[state]
            return stats.t.rvs(df_t, loc=loc, scale=scale, size=n,
                               random_state=rng.integers(0, 2**31))

    def simulate(self, current_regime: str, horizon: int,
                 n_paths: int, seed: int = 42) -> np.ndarray:
        rng    = np.random.default_rng(seed)
        P_vals = self.P.values
        cur_idx = self.states.index(current_regime)
        paths  = np.zeros((n_paths, horizon))

        for sim in range(n_paths):
            idx = cur_idx
            t   = 0
            while t < horizon:
                # Sample a block of returns for the current regime
                block_len = min(self.BLOCK_SIZE, horizon - t)
                block = self._sample_regime(self.states[idx], block_len, rng)
                paths[sim, t: t + block_len] = block
                t += block_len
                # Transition to next regime after the block
                idx = rng.choice(len(self.states), p=P_vals[idx])

        return paths   # shape: (n_paths, horizon)


class StationarityAnalyser:
    """
    Tests whether the Markov transition matrix is stationary over time.

    Two approaches implemented: 
    1. Rolling re-estimation: re-fit the transition matrix every STEP days
       on an expanding window and track how probabilities drift.
    2. KS Test: formally compare the transition probability distributions
       from the first half vs second half of the sample.
       A significant KS statistic means the matrix changed and hence, stationarity fails.
    """

    STEP = 63   # re-fit every quarter (63 trading days)

    def __init__(self, regimes: pd.Series, states: list):
        self.regimes = regimes.dropna()
        self.states  = states

    def rolling_matrices(self) -> list:
        """
        Returns list of (date, transition_matrix) tuples,
        re-fitted on expanding windows every STEP days.
        """
        results = []
        n = len(self.regimes)
        for end in range(252, n, self.STEP):   # need at least 1 year to start
            window = self.regimes.iloc[:end]
            eng = MarkovEngine(self.states)
            eng.fit(window)
            results.append((self.regimes.index[end - 1], eng.transition_matrix.copy()))
        return results

    def chi2_stationarity_test(self) -> pd.DataFrame:
        """
        Chi-squared test for homogeneity of transition matrices.

        The correct statistical test for Markov chain stationarity.
        Compares the TRANSITION COUNT matrices from two equal time periods.

        For each row (from-state), we test whether the distribution of
        transitions is the same in the first half vs the second half.

        H0: transition counts in period 1 and period 2 come from the
            same underlying distribution (stationarity holds).
        Reject H0 if p-value < 0.05.
        """
        n      = len(self.regimes)
        half   = n // 2
        first  = self.regimes.iloc[:half]
        second = self.regimes.iloc[half:]

        eng1 = MarkovEngine(self.states); eng1.fit(first)
        eng2 = MarkovEngine(self.states); eng2.fit(second)

        rows = []
        for s_from in self.states:
            # Build observed contingency table: rows = period, cols = to-state
            counts1 = [eng1.transition_counts.loc[s_from, s_to] for s_to in self.states]
            counts2 = [eng2.transition_counts.loc[s_from, s_to] for s_to in self.states]

            contingency = np.array([counts1, counts2], dtype=float)

            # Chi-squared test only valid if expected counts are sufficient
            row_totals = contingency.sum(axis=1, keepdims=True)
            if row_totals.min() < 5:
                chi2_stat, p_val = np.nan, np.nan
                verdict = "⚠️ Too few observations"
            else:
                try:
                    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency)
                    verdict = "❌ Non-stationary" if p_val < 0.05 else "✅ Stationary"
                except Exception:
                    chi2_stat, p_val = np.nan, np.nan
                    verdict = "⚠️ Could not compute"

            for s_to in self.states:
                p1 = eng1.transition_matrix.loc[s_from, s_to]
                p2 = eng2.transition_matrix.loc[s_from, s_to]
                rows.append({
                    "From → To":       f"{s_from} → {s_to}",
                    "P₁ (first half)": round(p1, 4),
                    "P₂ (second half)": round(p2, 4),
                    "Δ":               round(p2 - p1, 4),
                    "Row χ² stat":     round(chi2_stat, 3) if not np.isnan(chi2_stat) else "—",
                    "p-value":         round(p_val, 4) if not np.isnan(p_val) else "—",
                    "Verdict":         verdict,
                })

        return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    tickers_list = list(tickers)
    data = yf.download(tickers_list, start=start, end=end,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers_list[0]})
    return prices.dropna(how="all")

COLORS = {
    "Bull":    "#2ecc71",
    "Neutral": "#f39c12",
    "Bear":    "#e74c3c",
}
DEFAULT_COLOR = "#00b4d8"
STATE_LIST    = ["Bear", "Neutral", "Bull"]

st.set_page_config(
    page_title="MarkovQuant Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.35rem; }
.stTab [data-baseweb="tab"]   { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Engine Configuration")

st.sidebar.subheader("Asset Universe")
asset_class = st.sidebar.selectbox("Asset Class", ["Equities", "ETFs", "Crypto"])

PRESETS = {
    "Equities": ["SPY", "AAPL", "MSFT", "GOOGL", "NVDA"],
    "ETFs":     ["SPY", "QQQ", "IWM", "GLD", "TLT"],
    "Crypto":   ["BTC-USD", "ETH-USD", "SOL-USD"],
}

pool = PRESETS[asset_class]

tickers = st.sidebar.multiselect("Select Tickers", pool, default=pool[:1])
if not tickers:
    st.warning("Select at least one ticker in the sidebar.")
    st.stop()

primary = st.sidebar.selectbox("Primary Asset", tickers)

st.sidebar.subheader("Data Period")
start = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=5 * 365))
end   = st.sidebar.date_input("End Date",   datetime.today())

st.sidebar.subheader("Risk Parameters")
conf        = st.sidebar.slider("VaR Confidence", 0.90, 0.99, 0.95, step=0.01)
fwd_horizon = st.sidebar.slider("Forward Horizon (months)", 1, 24, 6)
train_pct   = st.sidebar.slider("Backtest Train Split", 0.50, 0.85, 0.70)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Regime detection:** Gaussian HMM (3 latent states)\n\n"
    "**Monte Carlo:** Historical Block Bootstrap\n\n"
    "**Backtesting:** Vol-scaled friction model\n\n"
    "**VaR validation:** Kupiec POF Test"
)

st.title("MarkovQuant: Market Regime Risk Engine")
st.markdown(
    "Hidden Markov Model regime detection · Regime-conditional VaR/CVaR · "
    "Kupiec validation · Friction-aware backtesting · Block Bootstrap Monte Carlo"
)
st.divider()

with st.spinner(f"Fetching data for {', '.join(tickers)}…"):
    prices = load_prices(tuple(tickers), str(start), str(end))

if prices.empty:
    st.error("No price data returned. Check tickers and date range.")
    st.stop()

primary_prices = prices[primary].dropna()
returns        = primary_prices.pct_change().dropna()

with st.spinner("Fitting HMM regimes (EM algorithm)…"):
    classifier = RegimeClassifier()
    regimes    = classifier.classify(primary_prices).dropna()

aligned = pd.DataFrame({
    "price":  primary_prices,
    "return": returns,
    "regime": regimes,
}).dropna()

state_list     = [s for s in STATE_LIST if s in aligned["regime"].unique()]
engine         = MarkovEngine(state_list)
engine.fit(aligned["regime"])
P              = engine.transition_matrix
risk_engine    = RiskEngine(aligned["return"], aligned["regime"])
current_regime = aligned["regime"].iloc[-1]

bt               = Backtester(primary_prices, regimes, train_pct=train_pct)
test_df, _       = bt.run()
split_date       = test_df.index[0]

train_mask          = aligned.index < split_date
train_risk_engine   = RiskEngine(
    aligned[train_mask]["return"],
    aligned[train_mask]["regime"],   # full-sample HMM labels on train period
)
oos_returns         = test_df["r"]
oos_regimes_series  = test_df["regime"]   # train-only HMM labels on test period

def calc_dd(rets: pd.Series) -> pd.Series:
    """Compute drawdown series from a return series."""
    cum = (1 + rets).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Regimes",
    "Transition Matrix",
    "Risk Analytics",
    "Backtesting",
    "Forward Projections & Monte Carlo",
    "Stationarity Analysis",
])

# TAB 1 – MARKET REGIMES

with tab1:
    st.header("HMM Market Regime Detection")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Regime", current_regime)
    c2.metric("Total Observations", len(aligned))
    c3.metric("HMM States", len(state_list))
    ann_vol = returns.tail(21).std() * np.sqrt(252)
    c4.metric("21-day Ann. Vol", f"{ann_vol:.1%}")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"{primary} — Price with HMM Regime Overlay", "Daily Returns (%)"],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(x=aligned.index, y=aligned["price"],
                   mode="lines", name="Price",
                   line=dict(color="#00b4d8", width=1.5)),
        row=1, col=1,
    )
    for reg in state_list:
        mask = aligned["regime"] == reg
        fig.add_trace(
            go.Scatter(
                x=aligned[mask].index, y=aligned[mask]["price"],
                mode="markers",
                marker=dict(size=3, color=COLORS.get(reg, DEFAULT_COLOR)),
                name=reg, legendgroup=reg,
            ),
            row=1, col=1,
        )
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in aligned["return"]]
    fig.add_trace(
        go.Bar(x=aligned.index, y=aligned["return"] * 100,
               marker_color=bar_colors, name="Daily Return", showlegend=False),
        row=2, col=1,
    )
    fig.update_layout(
        height=620, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        counts = aligned["regime"].value_counts()
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            title="Regime Distribution",
            color=counts.index, color_discrete_map=COLORS,
            template="plotly_dark",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Regime Duration Statistics")
        grp_id = (aligned["regime"] != aligned["regime"].shift()).cumsum()
        dur = aligned.groupby(grp_id)["regime"].agg(["first", "count"])
        dur_stats = (
            dur.groupby("first")["count"]
            .agg(["mean", "median", "max", "min"])
            .rename(columns={"mean": "Avg Days", "median": "Median",
                              "max": "Longest", "min": "Shortest"})
            .round(1)
        )
        st.dataframe(dur_stats, use_container_width=True)
        st.caption(
            "Longer average durations indicate more persistent regimes — "
            "a property the HMM explicitly models through its transition structure."
        )

# TAB 2 – TRANSITION MATRIX

with tab2:
    st.header("Markov Chain — Transition Matrix")

    with st.expander("MLE Estimation"):
        st.latex(r"\hat{P}_{ij} = \frac{N_{ij}}{\sum_k N_{ik}}")
        st.markdown(
            "$N_{ij}$ = observed transitions from regime $i$ to $j$. "
            "Divide by row total to get probabilities. Each row sums to 1."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Transition Probability Matrix")
        st.dataframe(
            P.round(4).style.background_gradient(cmap="RdYlGn", axis=1),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Observed Transition Counts")
        st.dataframe(engine.transition_counts.astype(int), use_container_width=True)

    fig_heat = px.imshow(
        P.values, x=state_list, y=state_list,
        color_continuous_scale="RdYlGn", text_auto=".3f",
        labels={"x": "To", "y": "From", "color": "Prob"},
        title="Transition Probability Heatmap",
        template="plotly_dark",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Stationary Distribution  π")
    with st.expander("Long-Run Equilibrium"):
        st.latex(r"\pi P = \pi, \quad \sum_i \pi_i = 1")
        st.markdown(
            "In the long run, the market spends fraction $\\pi_i$ of time in regime $i$, "
            "regardless of starting state — if the chain is ergodic."
        )

    try:
        pi  = engine.stationary_distribution()
        mrt = engine.mean_recurrence_time()
        c1, c2 = st.columns(2)
        with c1:
            fig_pi = px.bar(
                x=pi.index, y=pi.values,
                title="Long-Run Regime Probabilities",
                labels={"x": "Regime", "y": "π"},
                color=pi.index, color_discrete_map=COLORS,
                template="plotly_dark",
            )
            st.plotly_chart(fig_pi, use_container_width=True)
        with c2:
            mrt_df = pd.DataFrame({
                "Stationary Prob (π)": pi.round(4),
                "Mean Recurrence (days)": mrt.round(1),
            })
            st.subheader("Mean Recurrence Time")
            st.caption("Expected trading days to return to each regime once left.")
            st.dataframe(mrt_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Stationary distribution error: {e}")

    st.subheader("Chapman–Kolmogorov N-Step Calculator")
    with st.expander("N-step transitions"):
        st.latex(r"P^{(n)}_{ij} = \left(P^n\right)_{ij}")
        st.markdown("Probability of moving from regime $i$ to $j$ in exactly $n$ trading days.")

    c1, c2, c3 = st.columns(3)
    with c1: from_s  = st.selectbox("From Regime", state_list, key="ck_from")
    with c2: to_s    = st.selectbox("To Regime",   state_list, key="ck_to")
    with c3: n_steps = st.number_input("Steps (days)", 1, 500, 21)

    try:
        prob_n = engine.n_step_matrix(n_steps).loc[from_s, to_s]
        st.metric(f"P({from_s} → {to_s} | {n_steps} days)", f"{prob_n:.4f}")

        max_steps  = min(int(n_steps) + 1, 252)
        convergence = [engine.n_step_matrix(s).loc[from_s, to_s] for s in range(1, max_steps)]
        fig_conv   = go.Figure(go.Scatter(
            x=list(range(1, max_steps)), y=convergence,
            mode="lines", line=dict(color="#00b4d8", width=2),
        ))
        fig_conv.add_hline(
            y=engine.stationary_distribution()[to_s],
            line_dash="dash", line_color="#f39c12",
            annotation_text="Stationary π",
        )
        fig_conv.update_layout(
            title=f"Convergence of P({from_s} → {to_s})",
            xaxis_title="Steps (days)", yaxis_title="Probability",
            template="plotly_dark",
        )
        st.plotly_chart(fig_conv, use_container_width=True)
    except Exception as e:
        st.error(str(e))

# TAB 3 – RISK ANALYTICS

with tab3:
    st.header("Regime-Conditional Risk Analytics")

    alpha       = 1 - conf
    overall_var = risk_engine.var(alpha)
    overall_cvar = risk_engine.cvar(alpha)
    ann_ret     = aligned["return"].mean() * 252
    sharpe      = aligned["return"].mean() / aligned["return"].std() * np.sqrt(252)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR {conf:.0%} (overall)",  f"{overall_var:.2%}")
    c2.metric(f"CVaR {conf:.0%} (overall)", f"{overall_cvar:.2%}")
    c3.metric("Ann. Return", f"{ann_ret:.2%}")
    c4.metric("Ann. Sharpe", f"{sharpe:.2f}")

    st.subheader("Risk Metrics by Regime")
    summary = risk_engine.regime_summary()
    st.dataframe(
        summary.style.background_gradient(
            cmap="RdYlGn", subset=["Sharpe (Ann)", "Mean Ret (Ann %)"]
        ),
        use_container_width=True,
    )

    # VaR / CVaR bar chart
    regime_vars  = {r: risk_engine.var(alpha, r)  for r in state_list}
    regime_cvars = {r: risk_engine.cvar(alpha, r) for r in state_list}

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        name=f"VaR {conf:.0%}", x=list(regime_vars.keys()),
        y=[v * 100 for v in regime_vars.values()],
        marker_color="#e74c3c",
    ))
    fig_risk.add_trace(go.Bar(
        name=f"CVaR {conf:.0%}", x=list(regime_cvars.keys()),
        y=[v * 100 for v in regime_cvars.values()],
        marker_color="#c0392b",
    ))
    fig_risk.update_layout(
        barmode="group",
        title=f"VaR & CVaR at {conf:.0%} Confidence — by HMM Regime",
        xaxis_title="Regime", yaxis_title="Daily Loss (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Return distributions
    st.subheader("Return Distribution by Regime")
    fig_viol = go.Figure()
    for reg in state_list:
        sub = aligned[aligned["regime"] == reg]["return"] * 100
        fig_viol.add_trace(go.Violin(
            y=sub, name=reg,
            box_visible=True, meanline_visible=True,
            fillcolor=COLORS.get(reg, DEFAULT_COLOR),
            opacity=0.7, line_color="white",
        ))
    fig_viol.update_layout(
        title="Empirical Return Distributions per HMM Regime",
        yaxis_title="Daily Return (%)", template="plotly_dark",
    )
    st.plotly_chart(fig_viol, use_container_width=True)

    # Rolling VaR
    st.subheader(f"Rolling 30-Day VaR ({conf:.0%})")
    roll_var = aligned["return"].rolling(30).quantile(alpha) * 100
    fig_rvar = go.Figure(go.Scatter(
        x=roll_var.index, y=roll_var,
        fill="tozeroy", name="Rolling VaR",
        line=dict(color="#e74c3c"),
    ))
    fig_rvar.update_layout(
        xaxis_title="Date", yaxis_title="VaR (%)", template="plotly_dark",
    )
    st.plotly_chart(fig_rvar, use_container_width=True)

    # ── Kupiec Test  —  out-of-sample only
    st.subheader("Kupiec Test — Out-of-Sample VaR Validation")
    with st.expander("What is the Kupiec Test?"):
        st.markdown(r"""
The **Kupiec Proportion of Failures (POF) Test** formally validates whether our VaR
model is correctly calibrated. It answers the question if the observed breach rate matches the claimed one?

$$LR = -2\ln\left(\frac{(1-p)^{T-N} \cdot p^N}{(1-\hat{p})^{T-N} \cdot \hat{p}^N}\right) \sim \chi^2_1$$

Reject H₀ (model is miscalibrated) if $LR > 3.84$.
        """)

    # Kupiec uses pre-computed OOS data from the single backtester run above.
    # train_risk_engine and oos_returns/oos_regimes_series are already available.
    st.markdown(
        f"VaR estimated on **training data** (before {split_date.date()}). "
        f"Kupiec evaluated on **{len(oos_returns)} out-of-sample days** after that date."
    )

    st.markdown("#### Overall OOS")
    # Using train VaR threshold, count breaches in OOS returns
    train_var_overall = train_risk_engine.var(alpha)
    N_oos   = int((oos_returns < train_var_overall).sum())
    T_oos   = len(oos_returns)
    p_hat   = N_oos / T_oos if T_oos > 0 else 0

    if T_oos >= 30 and p_hat not in (0.0, 1.0):
        LR_oos = -2 * (
            np.log(((1 - alpha) ** (T_oos - N_oos)) * (alpha ** N_oos)) -
            np.log(((1 - p_hat) ** (T_oos - N_oos)) * (p_hat ** N_oos))
        )
        pv_oos   = 1 - stats.chi2.cdf(LR_oos, df=1)
        rejected = LR_oos > 3.84
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OOS Days (T)",      T_oos)
        c2.metric("Actual Breaches",   N_oos)
        c3.metric("Expected Breaches", round(T_oos * alpha, 1))
        c4.metric("Breach Rate",       f"{p_hat:.2%}")
        c1, c2 = st.columns(2)
        c1.metric("LR Statistic", f"{LR_oos:.4f}",
                  delta="Critical value: 3.84", delta_color="off")
        c2.metric("p-value", f"{pv_oos:.4f}")
        st.info("❌ REJECTED — VaR miscalibrated OOS" if rejected
                else "✅ NOT REJECTED — VaR well calibrated OOS")
    else:
        st.warning("Insufficient OOS observations for Kupiec test. Extend date range or reduce train split.")

    st.markdown("#### Christoffersen Conditional Coverage Test — Overall OOS")
    with st.expander("Why Christoffersen and not just Kupiec?"):
        st.markdown(r"""
**Kupiec's limitation:** it only counts total breaches. A model that produces
10 consecutive VaR breaches during a crash, then none for months, has the right
*total count* and passes Kupiec, but it's clearly wrong. It gave no warning.

**Christoffersen (1998)** adds an independence test on top of Kupiec:

$$LR_{ind} = -2\ln\frac{L(\hat{p})}{L(\hat{p}_{01}, \hat{p}_{11})} \sim \chi^2_1$$

Where $\hat{p}_{01}$ = breach probability after a calm day, $\hat{p}_{11}$ = breach
probability after a breach day. If $\hat{p}_{11} \gg \hat{p}_{01}$, violations are
clustering — the model is failing silently during crises.

The combined **Conditional Coverage** statistic:
$$LR_{cc} = LR_{pof} + LR_{ind} \sim \chi^2_2 \quad \text{(critical value: 5.99)}$$

A model must pass **both** frequency AND independence to be well-calibrated.
        """)

    hits_oos = (oos_returns.values < train_var_overall).astype(int)
    cc = RiskEngine.christoffersen_test(hits_oos, alpha)

    if "error" not in cc:
        st.markdown("**Breach transition counts** — tells you if violations cluster:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("No breach → No breach (n₀₀)", cc["n_00"])
        c2.metric("No breach → Breach (n₀₁)",    cc["n_01"])
        c3.metric("Breach → No breach (n₁₀)",    cc["n_10"])
        c4.metric("Breach → Breach (n₁₁)",       cc["n_11"])

        st.markdown(
            f"Breach prob after calm day: **{cc['p_01']:.2%}** · "
            f"Breach prob after breach day: **{cc['p_11']:.2%}**"
        )
        if cc["p_11"] > cc["p_01"] * 1.5:
            st.warning(
                f"⚠️ p₁₁ ({cc['p_11']:.2%}) is materially higher than "
                f"p₀₁ ({cc['p_01']:.2%}) — violations are clustering. "
                "The model underestimates risk during crises."
            )
        else:
            st.success("Breach probabilities are similar — violations are not clustering.")

        col1, col2, col3 = st.columns(3)
        col1.metric("LR_pof (Kupiec)",       f"{cc['LR_pof']:.4f}",
                    delta=f"p={cc['pv_pof']:.3f} · crit=3.84", delta_color="off")
        col2.metric("LR_ind (Independence)", f"{cc['LR_ind']:.4f}",
                    delta=f"p={cc['pv_ind']:.3f} · crit=3.84", delta_color="off")
        col3.metric("LR_cc  (Combined CC)",  f"{cc['LR_cc']:.4f}",
                    delta=f"p={cc['pv_cc']:.3f} · crit=5.99", delta_color="off")

        res = pd.DataFrame({
            "Test":           ["Kupiec POF", "Independence", "Conditional Coverage"],
            "LR Statistic":   [cc["LR_pof"], cc["LR_ind"], cc["LR_cc"]],
            "p-value":        [cc["pv_pof"], cc["pv_ind"], cc["pv_cc"]],
            "Critical Value": [3.84, 3.84, 5.99],
            "Verdict":        [cc["verdict_pof"], cc["verdict_ind"], cc["verdict_cc"]],
        }).set_index("Test")
        st.dataframe(res, use_container_width=True)

        if not cc["reject_cc"]:
            st.success("✅ PASSES Conditional Coverage — correct frequency AND independent violations.")
        elif cc["reject_pof"] and not cc["reject_ind"]:
            st.error("❌ Wrong breach frequency (Kupiec fails), but violations are independent.")
        elif not cc["reject_pof"] and cc["reject_ind"]:
            st.error("❌ Correct breach count but violations cluster — model fails during crises.")
        else:
            st.error("❌ Both frequency and independence rejected — model is poorly calibrated.")
    else:
        st.warning(cc["error"])


    kt_rows = []
    for reg in state_list:
        train_var_reg = train_risk_engine.var(alpha, reg)
        oos_reg_rets  = oos_returns[oos_regimes_series == reg]
        T_r = len(oos_reg_rets)
        if T_r < 30:
            kt_rows.append({"Regime": reg, "Verdict": f"⚠️ Only {T_r} OOS observations",
                            "T": T_r, "N": "—", "Expected": "—",
                            "Breach Rate": "—", "LR": "—", "p-value": "—"})
            continue
        N_r   = int((oos_reg_rets < train_var_reg).sum())
        ph_r  = N_r / T_r
        if ph_r in (0.0, 1.0):
            kt_rows.append({"Regime": reg, "T": T_r, "N": N_r,
                            "Expected": round(T_r * alpha, 1),
                            "Breach Rate": f"{ph_r:.2%}",
                            "LR": "—", "p-value": "—",
                            "Verdict": "⚠️ Edge case"})
            continue
        LR_r  = -2 * (
            np.log(((1 - alpha) ** (T_r - N_r)) * (alpha ** N_r)) -
            np.log(((1 - ph_r)  ** (T_r - N_r)) * (ph_r  ** N_r))
        )
        pv_r  = 1 - stats.chi2.cdf(LR_r, df=1)
        kt_rows.append({
            "Regime":       reg,
            "T":            T_r,
            "N":            N_r,
            "Expected":     round(T_r * alpha, 1),
            "Breach Rate":  f"{ph_r:.2%}",
            "LR":           round(LR_r, 4),
            "p-value":      round(pv_r, 4),
            "Verdict":      "❌ Rejected" if LR_r > 3.84 else "✅ Not Rejected",
        })
    if kt_rows:
        st.dataframe(pd.DataFrame(kt_rows).set_index("Regime"), use_container_width=True)

# TAB 4 – BACKTESTING

with tab4:
    st.header("Walk-Forward Backtesting — Friction-Aware")

    with st.expander("Strategy & friction model"):
        st.markdown(f"""
**Regime Momentum Strategy**
- Signal: +1 (long) in Bull, −1 (short) in Bear, 0 in Neutral
- Lagged by 1 day to prevent lookahead bias

**Vol-Scaled Transaction Cost Model**
- Base spread: {Backtester.BASE_SPREAD * 10000:.0f} bps per trade
- Scaled by `current_vol / mean_vol` — bid-ask spreads widen in high-vol regimes
- Charged every time the signal changes (regime flip = trade)
        """)

    # test_df already computed at load time — no second HMM fit needed
    cum_s  = (1 + test_df["strategy"]).cumprod()
    cum_sg = (1 + test_df["strategy_gross"]).cumprod()
    cum_b  = (1 + test_df["buy_hold"]).cumprod()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=test_df.index, y=cum_s,
        name="Strategy (net of costs)", line=dict(color="#00b4d8", width=2.5),
    ))
    fig_bt.add_trace(go.Scatter(
        x=test_df.index, y=cum_sg,
        name="Strategy (gross)", line=dict(color="#00b4d8", width=1.5, dash="dot"),
    ))
    fig_bt.add_trace(go.Scatter(
        x=test_df.index, y=cum_b,
        name="Buy & Hold", line=dict(color="#95a5a6", width=2, dash="dash"),
    ))
    fig_bt.update_layout(
        title=f"Out-of-Sample Cumulative Returns — {primary}",
        xaxis_title="Date", yaxis_title="Growth of $1",
        template="plotly_dark", height=420,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Net of Costs")
        for k, v in Backtester.metrics(test_df["strategy"].dropna()).items():
            st.metric(k, v)
    with c2:
        st.subheader("Gross (no costs)")
        for k, v in Backtester.metrics(test_df["strategy_gross"].dropna()).items():
            st.metric(k, v)
    with c3:
        st.subheader("Buy & Hold")
        for k, v in Backtester.metrics(test_df["buy_hold"].dropna()).items():
            st.metric(k, v)

    # Drawdown comparison
    st.subheader("Drawdown Comparison")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=test_df.index, y=calc_dd(test_df["strategy"]) * 100,
        fill="tozeroy", name="Strategy (net)", line=dict(color="#00b4d8"),
    ))
    fig_dd.add_trace(go.Scatter(
        x=test_df.index, y=calc_dd(test_df["buy_hold"]) * 100,
        fill="tozeroy", name="Buy & Hold", line=dict(color="#95a5a6"), opacity=0.5,
    ))
    fig_dd.update_layout(
        title="Drawdown (%)", xaxis_title="Date",
        yaxis_title="Drawdown (%)", template="plotly_dark",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# TAB 5 – FORWARD PROJECTIONS & MONTE CARLO

with tab5:
    st.header("Forward-Looking Projections & Block Bootstrap Monte Carlo")
    st.info(f"**Current HMM Regime:** `{current_regime}`")

    # Forward probability fan
    st.subheader("Regime Probability Evolution (Chapman–Kolmogorov)")
    max_days = fwd_horizon * 21
    horizons = list(range(1, max_days + 1, max(1, max_days // 60)))

    prob_evo = {s: [] for s in state_list}
    for h in horizons:
        row = engine.n_step_matrix(h).loc[current_regime]
        for s in state_list:
            prob_evo[s].append(row.get(s, 0))

    fig_fwd = go.Figure()
    for s in state_list:
        fig_fwd.add_trace(go.Scatter(
            x=horizons, y=prob_evo[s], name=s, mode="lines",
            line=dict(color=COLORS.get(s, DEFAULT_COLOR), width=2.5),
        ))
    fig_fwd.update_layout(
        title=f"Forward Regime Probabilities from '{current_regime}'",
        xaxis_title="Trading Days Ahead", yaxis_title="Probability",
        template="plotly_dark", yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_fwd, use_container_width=True)

    # Forward VaR table
    st.subheader("Probability-Weighted Forward VaR / CVaR")
    with st.expander("Forward VaR methodology"):
        st.markdown(r"""
In $h$ trading days we may be in any regime with probability given by $P^h$.
The forward VaR weights each regime's VaR by the probability of being in that regime:

$$\text{VaR}^{\text{fwd}}(h) = \sum_j P^h_{i_0, j} \cdot \text{VaR}^{(j)}$$

This accounts for possible regime transitions between now and the horizon
        """)
    alpha = 1 - conf
    fwd_rows = {}
    for months in [1, 3, 6, 12]:
        if months <= fwd_horizon:
            r = risk_engine.forward_var(current_regime, months * 21, P, alpha=alpha)
            fwd_rows[f"{months}M"] = {
                f"Weighted VaR {conf:.0%}":  f"{r['weighted_var']:.3%}",
                f"Weighted CVaR {conf:.0%}": f"{r['weighted_cvar']:.3%}",
            }
    if fwd_rows:
        st.dataframe(pd.DataFrame(fwd_rows).T, use_container_width=True)

    st.divider()

    # Monte Carlo
    st.subheader("Block Bootstrap Monte Carlo — Regime-Conditional Price Paths")

    c1, c2 = st.columns(2)
    with c1: n_paths  = st.slider("Number of Paths", 100, 3000, 1000, step=100)
    with c2: sim_days = st.slider("Horizon (trading days)", 21, 504, 126, step=21)

    if st.button("Run Block Bootstrap Simulation", type="primary"):
        with st.spinner(f"Simulating {n_paths:,} paths × {sim_days} days…"):
            mc    = MonteCarloSimulator(aligned["return"], aligned["regime"], P)
            paths = mc.simulate(current_regime, sim_days, n_paths)
            cum   = np.cumprod(1 + paths, axis=1)

            p5, p25, p50, p75, p95 = [
                np.percentile(cum, q, axis=0) for q in [5, 25, 50, 75, 95]
            ]
            days_ax = list(range(1, sim_days + 1))

            fig_mc = go.Figure()
            for i in range(min(150, n_paths)):
                fig_mc.add_trace(go.Scatter(
                    x=days_ax, y=cum[i], mode="lines",
                    line=dict(width=0.4, color="rgba(0,180,216,0.06)"),
                    showlegend=False,
                ))
            fig_mc.add_trace(go.Scatter(
                x=days_ax + days_ax[::-1],
                y=list(p95) + list(p5[::-1]),
                fill="toself", fillcolor="rgba(0,180,216,0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                name="5th–95th Pct",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days_ax + days_ax[::-1],
                y=list(p75) + list(p25[::-1]),
                fill="toself", fillcolor="rgba(0,180,216,0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                name="25th–75th Pct",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days_ax, y=p50, name="Median",
                line=dict(color="#00b4d8", width=2.5),
            ))
            fig_mc.add_trace(go.Scatter(
                x=days_ax, y=p5, name="5th Pct (sim VaR)",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            ))
            fig_mc.add_hline(y=1.0, line_dash="dot", line_color="white",
                             annotation_text="Initial Capital")
            fig_mc.update_layout(
                title=(
                    f"Block Bootstrap: {n_paths:,} Paths — "
                    f"{sim_days}d from '{current_regime}'"
                ),
                xaxis_title="Trading Days", yaxis_title="Cumulative Return",
                template="plotly_dark", height=540,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            terminal = cum[:, -1]
            fig_term = px.histogram(
                x=terminal, nbins=60,
                title=f"Terminal Wealth Distribution (Day {sim_days})",
                labels={"x": "Cumulative Return"},
                template="plotly_dark",
                color_discrete_sequence=["#00b4d8"],
            )
            fig_term.add_vline(x=np.percentile(terminal, 5),
                               line_dash="dash", line_color="#e74c3c",
                               annotation_text=f"5th %ile: {np.percentile(terminal,5):.2f}")
            fig_term.add_vline(x=np.median(terminal),
                               line_dash="dash", line_color="#f39c12",
                               annotation_text=f"Median: {np.median(terminal):.2f}")
            st.plotly_chart(fig_term, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median Return",  f"{np.median(terminal) - 1:.2%}")
            c2.metric("Sim VaR (5th)",  f"{np.percentile(terminal, 5) - 1:.2%}")
            c3.metric("Prob. of Loss",  f"{(terminal < 1).mean():.2%}")
            c4.metric("Expected Return",f"{terminal.mean() - 1:.2%}")

# TAB 6 – STATIONARITY ANALYSIS

with tab6:
    st.header("Is the Transition Matrix Stable Over Time?")

    with st.spinner("Computing rolling transition matrices…"):
        sa = StationarityAnalyser(aligned["regime"], state_list)
        rolling = sa.rolling_matrices()

    if len(rolling) < 2:
        st.warning("Not enough data for stationarity analysis. Use a longer date range.")
    else:
        # Rolling probability chart for each transition
        st.subheader("Rolling Transition Probability Drift")
        st.caption(
            f"Each line shows how P(from → to) evolves as more data is included. "
            f"Re-fitted every {StationarityAnalyser.STEP} trading days on an expanding window."
        )

        dates = [r[0] for r in rolling]

        # One chart per "from" state
        for s_from in state_list:
            fig_roll = go.Figure()
            for s_to in state_list:
                probs = [r[1].loc[s_from, s_to] if s_from in r[1].index and s_to in r[1].columns
                         else 0 for r in rolling]
                fig_roll.add_trace(go.Scatter(
                    x=dates, y=probs, name=f"→ {s_to}",
                    mode="lines",
                    line=dict(color=COLORS.get(s_to, DEFAULT_COLOR), width=2),
                ))
            fig_roll.update_layout(
                title=f"P({s_from} → ?) over time",
                xaxis_title="Date", yaxis_title="Transition Probability",
                yaxis_range=[0, 1], template="plotly_dark", height=300,
            )
            st.plotly_chart(fig_roll, use_container_width=True)

        # KS drift table
        st.subheader("Chi-Squared Homogeneity Test — First Half vs Second Half")
        st.caption(
            "Compares transition count distributions from the first 50% of data "
            "vs the last 50% using a chi-squared test for homogeneity. "
            "A p-value < 0.05 means the transition probabilities changed significantly — "
            "stationarity is violated for that from-state."
        )
        
        chi2_df = sa.chi2_stationarity_test()
        st.dataframe(
            chi2_df.style.apply(
                lambda col: [
                    "background-color: rgba(231,76,60,0.3)" if "Non-stationary" in str(v)
                    else "background-color: rgba(46,204,113,0.2)" if "Stationary" in str(v)
                    else "" for v in col
                ],
                subset=["Verdict"]
            ),
            use_container_width=True,
        )

        st.markdown("""
**How to interpret this:**
- **✅ Stationary** — chi-squared test does not reject homogeneity. Transition
  probabilities are statistically consistent across both halves of the sample.
- **❌ Non-stationary** — p-value < 0.05. The transition structure changed
  significantly between the two periods. For this from-state, the model's
  assumption of constant transition probabilities is statistically violated.
- **⚠️ Too few observations** — insufficient counts for a reliable test.
        """)

    st.divider()
    st.caption(
        "MarkovQuant | HMM · Kupiec · Block Bootstrap · Friction-Aware Backtesting · "
        "Stationarity Analysis"
    )
