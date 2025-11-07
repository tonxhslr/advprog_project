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


def price_american(option_type, expiration, s_0, k_0, iv, r, q, q_interval, nr_of_simulations, nr_of_timesteps):
    pass
