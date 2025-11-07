# Module-level constant for year days (consistent with calculate_expiration)
YEAR_DAYS_DEFAULT = 365.25  # keep consistent with calculate_expiration
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
import string
import json
import csv
import os
from datetime import date, datetime, timedelta

# Analytical Solution for European Option Price (Black-Scholes)
def black_scholes(S_0, K, T, r, sigma, option_type='call', q=0.0):
    '''
    Analytical solution (Black–Scholes–Merton form) for European options with optional
    continuous dividend yield q (q=0 reduces to the original no-dividend case).
    '''
    option_type = option_type.lower()

    if T <= 0 or sigma <= 0:
        intrinsic = max(S_0 - K, 0.0) if option_type == 'call' else max(K - S_0, 0.0)
        return float(intrinsic)

    # Helper terms with dividend yield q
    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        p = S_0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        p = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError('Option type must be "Call" or "Put"')

    return float(p)

# Monte-Carlo Simulation (supports discrete cash dividends on a fixed day interval)

def timestep_no_yield(start_price, r, sigma, delta_t):
    """
    One GBM step under the risk-neutral measure with drift r and NO continuous dividend yield.
    This is used between discrete, ex-dividend dates.
    """
    return start_price * np.exp((r - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * np.random.normal(0, 1))


def _dividend_steps(timesteps, T_years, q_interval_days, year_days=YEAR_DAYS_DEFAULT):
    """
    Compute the set of step indices (1..timesteps) that coincide with ex-dividend dates,
    when cash dividends are paid every q_interval_days.
    """
    if not q_interval_days or q_interval_days <= 0:
        return set()
    dt = T_years / timesteps
    interval_years = q_interval_days / year_days
    # dividend times in years (exclude t=0); map to nearest step index
    div_times = np.arange(interval_years, T_years + 1e-12, interval_years)
    div_steps = {int(round(t / dt)) for t in div_times}
    # keep only valid indices within 1..timesteps
    return {k for k in div_steps if 1 <= k <= timesteps}


def simulation_run(timesteps, start_price, expiration, r, q, sigma, q_interval_days=0, year_days=YEAR_DAYS_DEFAULT):
    """
    Simulate one asset path over `timesteps` up to maturity `expiration` (years).
    Discrete cash dividends of size `q` are paid every `q_interval_days`.
    Between dividends we use GBM with drift r (no continuous yield); at each ex-div step we drop:
        S <- max(S - q, 0)
    Returns: list of [step_index, price].
    """
    prices = [[0, start_price]]
    T = float(expiration)
    dt = T / timesteps
    S = start_price

    div_steps = _dividend_steps(timesteps, T, q_interval_days, year_days)

    for k in range(1, timesteps + 1):
        # evolve between dividend dates (risk-neutral drift r, no yield)
        S = timestep_no_yield(S, r, sigma, dt)
        # apply discrete cash dividend at ex-div date
        if k in div_steps and q is not None and q != 0:
            S = max(S - float(q), 0.0)
        prices.append([k, S])

    return prices


def MonteCarlo(simulations, timesteps, start_price, expiration, r, q, sigma, q_interval_days=0, year_days=YEAR_DAYS_DEFAULT):
    """
    Monte Carlo simulation of `simulations` paths, each with `timesteps` steps.
    If `q_interval_days` > 0, treats `q` as a CASH dividend amount paid every `q_interval_days` days.
    If `q_interval_days` == 0, no discrete cash dividends are applied.
    """
    return [
        simulation_run(timesteps, start_price, expiration, r, q, sigma, q_interval_days=q_interval_days, year_days=year_days)
        for _ in range(simulations)
    ]


# --- LSMC helpers: build exercise grid and simulate only on that grid ---

def _exercise_steps_and_times(expiration, timesteps_fine, q_interval_days=0, year_days=YEAR_DAYS_DEFAULT, exercise_every_days=1):
    """
    Build a set of exercise step indices (including 0 and the final step),
    using a daily grid by default and inserting nodes immediately *before* each ex-div step.

    Returns:
        exercise_steps: sorted list of unique step indices (0..timesteps_fine)
        t_grid_years: numpy array of times (in years) corresponding to exercise_steps
    """
    T = float(expiration)
    dt = T / timesteps_fine

    # Daily exercise grid (or every `exercise_every_days`)
    step_days = T * year_days / timesteps_fine
    stride = max(int(round(exercise_every_days / step_days)), 1)
    daily_steps = set(range(0, timesteps_fine + 1, stride))

    # Dividend steps (cash) according to q_interval_days, and also add the *day before* to allow exercise just before ex-div
    div_steps = _dividend_steps(timesteps_fine, T, q_interval_days, year_days)
    pre_div_steps = {max(k - 1, 0) for k in div_steps}

    # Union all and ensure 0 and final
    exercise_steps = sorted(daily_steps | div_steps | pre_div_steps | {0, timesteps_fine})
    t_grid_years = np.array([s * dt for s in exercise_steps], dtype=float)
    return exercise_steps, t_grid_years


def MonteCarlo_on_exercise_grid(simulations, timesteps_fine, start_price, expiration, r, q_cash, sigma, q_interval_days=0, year_days=YEAR_DAYS_DEFAULT, exercise_every_days=1):
    """
    Simulate paths with discrete cash dividends and return only prices on a coarser
    exercise grid suitable for LSMC (e.g., daily).  The underlying is simulated
    on `timesteps_fine`; dividends of CASH amount `q_cash` are subtracted on each ex-div step.

    Returns:
        S_grid: ndarray shape (M, K+1) prices at exercise times
        t_grid: ndarray shape (K+1,) exercise times (years)
    """
    T = float(expiration)
    dt = T / timesteps_fine
    exercise_steps, t_grid = _exercise_steps_and_times(T, timesteps_fine, q_interval_days, year_days, exercise_every_days)
    exercise_steps_set = set(exercise_steps)

    # Precompute dividend steps on the fine grid
    div_steps = _dividend_steps(timesteps_fine, T, q_interval_days, year_days)

    M = int(simulations)
    Kp1 = len(exercise_steps)
    S_grid = np.empty((M, Kp1), dtype=float)

    for m in range(M):
        S = start_price
        k_out = 0
        # record at t0
        S_grid[m, k_out] = S
        k_out += 1
        for k in range(1, timesteps_fine + 1):
            # evolve one fine step with drift r, no yield
            S = timestep_no_yield(S, r, sigma, dt)
            # ex-div cash drop
            if k in div_steps and q_cash is not None and q_cash != 0:
                S = max(S - float(q_cash), 0.0)
            # record if this is an exercise step
            if k in exercise_steps_set:
                S_grid[m, k_out] = S
                k_out += 1
    return S_grid, t_grid


# Testrun
# print(MonteCarlo(10, 10, 100, 1/12, 0.04, 0.01, 0.30))


# Reading / Transforming Inputs from File
def read_input_file(filepath):
    """
    Reads configuration parameters for the option pricing simulation
    from either a .json or .csv file.

    Returns: dict of parameters
    """

    ext = os.path.splitext(filepath)[1].lower()

    # JSON input
    if ext == ".json":
        with open(filepath, "r") as f:
            config = json.load(f)

    # CSV input
    elif ext == ".csv":
        config = {}
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # skip empty lines or comments
                if not row or row[0].startswith("#"):
                    continue
                # CSV is assumed to be key,value pairs
                key = row[0].strip()
                if len(row) > 1:
                    value = row[1].strip()
                else:
                    value = None
                config[key] = value

    else:
        raise ValueError("Unsupported file type. Please provide .json or .csv")

    return config


def calculate_expiration(start_date, start_time, expiration_date, expiration_time):
    '''
    Calculates time until expiration as a fraction of a year, given the start/end dates and times. 
    '''

    start = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
    if expiration_time == "AM":
        end_time = "09:00"
    elif expiration_time == "PM":
        end_time = "16:00"
    else:
        raise ValueError("Expiration must be either AM or PM")
    end = datetime.strptime(f"{expiration_date} {end_time}", "%Y-%m-%d %H:%M")

    return ((end-start).total_seconds())/(365.25*24*60*60)


def transform_input(file):
    '''
    Transforms all the inputs from a file into actual usable variables for the model

    Returns: Dictionary of inputs
    '''

    params = {}

    config = read_input_file(file)

    params["option_type"] = config["option_type"].lower()
    params["exercise_type"] = config["exercise_type"].lower()
    params["expiration"] = calculate_expiration(config["start_date"], config["start_time"], config["expiration_date"], config["expiration_time"])
    params["s_0"] = float(config["underlying_price"])
    params["k_0"] = float(config["option_strike"])
    params["iv"] = float(config["volatility"])/100
    params["r"] = float(config["interest_rate"])/100
    if config["dividends"] == "None" or config["dividends"] is False:
        params["q"] = 0
        params["q_interval"] = 0
    elif config["dividends"].lower() == "dividend":
        params["q"] = float(config["dividend_amount"])/float(config["underlying_price"]) # Because we want dividend to be a yield (if it's not a dividend stream)
        params["q_interval"] = 0
    elif config["dividends"].lower() == "dividend_stream":
        params["q"] = float(config["dividend_amount"])
        params["q_interval"] = float(config["day_interval"])
    else:
        raise ValueError("Please enter Dividend")
    params["nr_of_simulations"] = int(config["nr_simulations"])
    params["nr_of_timesteps"] = int(config["nr_of_timesteps"])
    if bool(config["output_to_file"]) == False: 
        params["filename"] = None
    elif bool(config["output_to_file"]) == True: 
        params["filename"] = config["output_filename"]
    else: 
        raise ValueError("Please enter an option for Output To File")
    
    if config["exercise_type"].lower() == "barrier":
        params["barrier_type"] = config["barrier_type"]
        params["threshold"] = config["threshold"]
    
    if config["exercise_type"].lower() == "binary":
        params["threshold"] = config["threshold"]
        params["binary_payout"] = config["binary_payout"]
    
    return params


def price_european(option_type, expiration, s_0, k_0, iv, r, q, q_interval, nr_of_simulations, nr_of_timesteps):
    '''
    Prices a European option under two regimes:
      1) If dividends are paid on a fixed day interval (q_interval > 0): use Monte Carlo.
      2) If no dividends (q == 0) or dividends are modeled as a continuous yield (q_interval == 0): use analytical Black–Scholes with dividend yield q.
    Returns: option price (float).
    '''
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option Type needs to be either Put or Call")

    T = float(expiration)
    sigma = float(iv)

    # Case 1: discrete periodic dividends -> Monte Carlo
    if q_interval and float(q_interval) > 0:
        paths = MonteCarlo(nr_of_simulations, nr_of_timesteps, s_0, T, r, q, sigma, q_interval_days=q_interval, year_days=YEAR_DAYS_DEFAULT)
        # each path is [[step, price], ...]; take the terminal price element [ -1 ][1]
        final_prices = [path[-1][1] for path in paths]
        if opt == "call":
            payoffs = np.maximum(np.array(final_prices) - k_0, 0.0)
        else:
            payoffs = np.maximum(k_0 - np.array(final_prices), 0.0)
        return float(np.exp(-r * T) * payoffs.mean())

    # Case 2: no dividends or continuous dividend yield -> analytical
    return black_scholes(s_0, k_0, T, r, sigma, opt, q=float(q))


def price_american(option_type, expiration, s_0, k_0, iv, r, q, q_interval, nr_of_simulations, nr_of_timesteps, exercise_every_days=1):
    """
    Price an American option using Least-Squares Monte Carlo (Longstaff–Schwartz).

    Args mirror the European pricer; `q` is interpreted as CASH dividend per payment
    when `q_interval > 0` (discrete stream). If `q_interval == 0`, there is no discrete cash drop
    and early exercise will almost never be optimal for calls (no-dividend case).

    The exercise grid is daily by default (exercise_every_days=1). Dividends are modeled
    as cash jumps on the fine grid; the grid also includes the step just *before* each ex-div date.
    """
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option Type needs to be either Put or Call")

    T = float(expiration)
    sigma = float(iv)

    # 1) Simulate only on the exercise grid (daily + pre-div nodes)
    S_grid, t_grid = MonteCarlo_on_exercise_grid(
        simulations=nr_of_simulations,
        timesteps_fine=nr_of_timesteps,
        start_price=s_0,
        expiration=T,
        r=r,
        q_cash=q if (q_interval and float(q_interval) > 0) else 0.0,
        sigma=sigma,
        q_interval_days=q_interval if q_interval else 0,
        year_days=YEAR_DAYS_DEFAULT,
        exercise_every_days=exercise_every_days,
    )

    # 2) Backward induction via regression
    M, Kp1 = S_grid.shape
    K = Kp1 - 1
    dt_step = np.diff(t_grid)  # length K

    # Payoff function
    if opt == 'call':
        payoff = lambda S: np.maximum(S - k_0, 0.0)
    else:
        payoff = lambda S: np.maximum(k_0 - S, 0.0)

    # Value at maturity (at t_K)
    V = payoff(S_grid[:, -1])  # shape (M,)

    # Work backwards k = K-1, ..., 0
    for k in range(K - 1, -1, -1):
        S_k = S_grid[:, k]
        intrinsic = payoff(S_k)
        disc = np.exp(-r * dt_step[k])

        # Discounted next-step value (realized continuation if we do not exercise at k)
        Y = disc * V

        # ITM paths for regression (classic LSMC); require at least 2-3 points
        itm = intrinsic > 0
        n_itm = int(np.count_nonzero(itm))
        cont_pred = np.zeros_like(S_k)

        if n_itm >= 3:
            # Basis on normalized price x = S/K
            x = (S_k[itm] / k_0)
            X = np.column_stack([np.ones_like(x), x, x**2])
            beta, *_ = np.linalg.lstsq(X, Y[itm], rcond=None)

            # Predict continuation for all paths
            x_all = (S_k / k_0)
            X_all = np.column_stack([np.ones_like(x_all), x_all, x_all**2])
            cont_pred = X_all @ beta
        else:
            # Not enough ITM samples to regress: default to continue
            cont_pred = np.full_like(S_k, Y.mean())

        # Exercise decision pathwise
        exercise_now = intrinsic >= cont_pred
        # If exercise now, value is intrinsic; otherwise continue with discounted next value
        V = np.where(exercise_now, intrinsic, Y)

    # 3) At t0, V is the pathwise value; the price is its mean
    return float(V.mean())


def price_asian(option_type, expiration, s_0, k_0, iv, r, q, q_interval, nr_of_simulations, nr_of_timesteps):
    '''
    Price a (European) Asian option via Monte Carlo using arithmetic averaging.

    Dividend handling:
      - If q_interval > 0: interpret `q` as CASH dividend per payment, applied every q_interval days.
        Paths are simulated on a fine grid with discrete cash drops; we average prices over fine steps 1..N (exclude t0).
      - If q_interval == 0: interpret `q` as a CONTINUOUS dividend yield; simulate GBM with drift (r - q).

    Averaging convention:
      - Arithmetic average over the simulated observation points at fine steps 1..N (exclude t0, include maturity).

    Returns: option price (float)
    '''
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option Type needs to be either Put or Call")

    T = float(expiration)
    sigma = float(iv)
    M = int(nr_of_simulations)
    N = int(nr_of_timesteps)

    if N <= 0:
        raise ValueError("nr_of_timesteps must be >= 1 for Asian averaging")

    dt = T / N

    # Case A: discrete CASH dividends on a fixed day interval -> use existing discrete-div simulator per path
    if q_interval and float(q_interval) > 0:
        avg_prices = np.empty(M, dtype=float)
        for m in range(M):
            path = simulation_run(N, s_0, T, r, q, sigma, q_interval_days=q_interval, year_days=YEAR_DAYS_DEFAULT)
            # path is [[step, price]] for steps 0..N; average over steps 1..N (exclude t0)
            prices_only = [p[1] for p in path[1:]]
            avg_prices[m] = float(np.mean(prices_only))
        if opt == 'call':
            payoffs = np.maximum(avg_prices - k_0, 0.0)
        else:
            payoffs = np.maximum(k_0 - avg_prices, 0.0)
        return float(np.exp(-r * T) * payoffs.mean())

    # Case B: no discrete dividends -> vectorized GBM with continuous dividend yield q
    # Simulate M paths with N steps under risk-neutral drift (r - q)
    Z = np.random.normal(size=(M, N))
    drift = (r - float(q) - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    # cumulative log-returns over time steps
    log_increments = drift + vol * Z
    log_levels = np.cumsum(log_increments, axis=1)
    S_paths = s_0 * np.exp(log_levels)  # shape (M, N), corresponds to steps 1..N

    # Arithmetic average across steps 1..N
    avg_prices = S_paths.mean(axis=1)

    if opt == 'call':
        payoffs = np.maximum(avg_prices - k_0, 0.0)
    else:
        payoffs = np.maximum(k_0 - avg_prices, 0.0)

    return float(np.exp(-r * T) * payoffs.mean())


def price_binary(option_type, expiration, s_0, k_0, iv, r, q, q_interval, nr_of_simulations, nr_of_timesteps):
    """
    Price a European **cash-or-nothing** binary option with unit payoff.

    Conventions:
      - `option_type`: 'call' pays 1 if S_T > K; 'put' pays 1 if S_T < K
      - `k_0` is the threshold/strike K
      - If `q_interval == 0`: interpret `q` as a **continuous dividend yield** and use the
        Black–Scholes closed-form for binaries (digital options).
      - If `q_interval > 0`: interpret `q` as **cash dividend per payment** every `q_interval` days
        and use Monte Carlo with discrete ex-div cash drops; payoff = 1{condition at T}.

    Returned price is for **unit payout**. If you need a payout Q, multiply the result by Q.
    """
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option Type needs to be either Put or Call")

    T = float(expiration)
    sigma = float(iv)
    K = float(k_0)

    # Edge cases
    if T <= 0 or sigma <= 0:
        if opt == 'call':
            payoff = 1.0 if s_0 > K else 0.0
        else:
            payoff = 1.0 if s_0 < K else 0.0
        return float(np.exp(-r * max(T, 0.0)) * payoff)

    # Case 1: analytic (continuous dividend yield q)
    if not q_interval or float(q_interval) == 0.0:
        d1 = (np.log(s_0 / K) + (r - float(q) + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt == 'call':
            return float(np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(np.exp(-r * T) * norm.cdf(-d2))

    # Case 2: discrete cash dividends -> Monte Carlo on terminal price
    paths = MonteCarlo(nr_of_simulations, nr_of_timesteps, s_0, T, r, q, sigma,
                       q_interval_days=q_interval, year_days=YEAR_DAYS_DEFAULT)
    S_T = np.array([path[-1][1] for path in paths], dtype=float)
    if opt == 'call':
        indicators = (S_T > K).astype(float)
    else:
        indicators = (S_T < K).astype(float)
    return float(np.exp(-r * T) * indicators.mean())


def price_barrier(option_type, barrier_type, barrier_price, expiration, s_0, k_0, iv, r, q, q_interval,
                  nr_of_simulations, nr_of_timesteps):
    """
    Monte-Carlo pricing for European barrier options (knock-in / knock-out).

    Conventions
    ----------
    - option_type: 'call' or 'put'
    - barrier_type: 'knock-in' or 'knock-out' (case-insensitive)
    - barrier direction is inferred:
        * if barrier_price > s_0  -> UP barrier (trigger when S >= B)
        * if barrier_price < s_0  -> DOWN barrier (trigger when S <= B)
        * if barrier_price == s_0 -> treated as already touched at t=0
    - Dividends:
        * if q_interval > 0: interpret q as CASH per payment every q_interval days (discrete drops via simulation_run)
        * else: interpret q as a CONTINUOUS dividend yield in GBM drift (r - q)

    Returns: present value (float) of the barrier option with **European** payoff at T.
    """
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    btype = barrier_type.strip().lower()
    if btype not in ("knock-in", "knock-out", "knockin", "knockout"):
        raise ValueError("barrier_type must be 'knock-in' or 'knock-out'")
    is_knock_in = (btype in ("knock-in", "knockin"))

    T = float(expiration)
    sigma = float(iv)
    B = float(barrier_price)
    M = int(nr_of_simulations)
    N = int(nr_of_timesteps)

    if N <= 0:
        raise ValueError("nr_of_timesteps must be >= 1 for barrier monitoring")

    # Determine barrier direction
    if B > s_0:
        # UP barrier, triggers on S >= B
        def hit_barrier(S_path_vals):
            return np.any(np.asarray(S_path_vals) >= B)
    elif B < s_0:
        # DOWN barrier, triggers on S <= B
        def hit_barrier(S_path_vals):
            return np.any(np.asarray(S_path_vals) <= B)
    else:
        # B == S0 -> considered already touched
        def hit_barrier(S_path_vals):
            return True

    # Vanilla payoff at maturity
    if opt == 'call':
        payoff_fn = lambda ST: max(ST - k_0, 0.0)
    else:
        payoff_fn = lambda ST: max(k_0 - ST, 0.0)

    disc = math.exp(-r * T)

    # Short-circuit edge cases at inception for KO/KI
    already_hit_at_start = hit_barrier([s_0])
    if already_hit_at_start:
        if is_knock_in:
            # Knock-in active immediately -> reduces to vanilla European pricing
            return disc * payoff_fn(s_0) if T <= 0 or sigma <= 0 else (
                price_european(opt, T, s_0, k_0, sigma, r, q, 0, 0, 0)
            )
        else:
            # Knock-out knocked out at inception
            return 0.0

    # --- Monte Carlo simulation and barrier monitoring ---
    payoffs = np.empty(M, dtype=float)

    if q_interval and float(q_interval) > 0:
        # Discrete CASH dividends case: use existing simulator that applies cash drops
        for m in range(M):
            path = simulation_run(N, s_0, T, r, q, sigma, q_interval_days=q_interval, year_days=YEAR_DAYS_DEFAULT)
            # Extract the price trajectory (including t0 and all steps)
            S_vals = [p[1] for p in path]  # length N+1
            touched = hit_barrier(S_vals)
            ST = S_vals[-1]
            vanilla = payoff_fn(ST)
            if is_knock_in:
                payoffs[m] = vanilla if touched else 0.0
            else:
                payoffs[m] = 0.0 if touched else vanilla
    else:
        # Continuous dividend yield case: simulate GBM with drift (r - q)
        dt = T / N
        mu = r - float(q)
        vol = sigma
        for m in range(M):
            S = s_0
            S_vals = [S]
            for k in range(1, N + 1):
                Z = np.random.normal()
                S = S * math.exp((mu - 0.5 * vol*vol) * dt + vol * math.sqrt(dt) * Z)
                S_vals.append(S)
            touched = hit_barrier(S_vals)
            ST = S_vals[-1]
            vanilla = payoff_fn(ST)
            if is_knock_in:
                payoffs[m] = vanilla if touched else 0.0
            else:
                payoffs[m] = 0.0 if touched else vanilla

    return float(disc * payoffs.mean())



def calculate_option_price(file):
    '''
    Given an input file with all necessary information, returns Option Price
    '''
    params = transform_input(file)

    type = params["option_type"]
    T = params["expiration"]
    s_0 = params["s_0"] 
    k_0 = params["k_0"]
    iv = params["iv"]
    r = params["r"]
    q = params["q"]
    q_interval = params["q_interval"]
    nr_sims = params["nr_of_simulations"]
    nr_steps = params["nr_of_timesteps"]

    if params["exercise_type"] == "european":
        p = price_european(type, T, s_0, k_0, iv, r, q, q_interval, nr_sims, nr_steps)
        
    elif params["exercise_type"] == "american":
        p = price_american(type, T, s_0, k_0, iv, r, q, q_interval, nr_sims, nr_steps)
        
    elif params["exercise_type"] == "asian":
        p = price_asian(type, T, s_0, k_0, iv, r, q, q_interval, nr_sims, nr_steps)
    
    elif params["exercise_type"] == "binary":
        p = price_binary(type, T, s_0, k_0, iv, r, q, q_interval, nr_sims, nr_steps)
        
    elif params["exercise_type"] == "barrier":
        p = price_barrier(type, params["barrier_type"], params["barrier_price"], T, s_0, k_0, iv, r, q, q_interval, nr_sims, nr_steps)

    return p
    

filename_csv = os.path.join(os.getcwd(), 'Group Project', 'try.csv')
print(calculate_option_price(filename_csv))
