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
        params["q"] = float(config["dividend_amount"])
        params["q_interval"] = 0
    elif config["dividends"].lower() == "dividend_stream":
        params["q"] = float(config["dividend_amount"])
        params["q_interval"] = float(config["day_interval"])
    else:
        raise ValueError("Please enter Dividend")
    params["nr_of_simulations"] = float(config["nr_simulations"])
    params["nr_of_timesteps"] = float(config["time_step"])
    if bool(config["output_to_file"]) == False: 
        params["filename"] = None
    elif bool(config["output_to_file"]) == True: 
        params["filename"] = config["output_filename"]
    else: 
        raise ValueError("Please enter an option for Output To File")
    
    if config["option_type"].lower() == "barrier":
        params["barrier_type"] = config["barrier_type"]
        params["threshold"] = config["threshold"]
    
    if config["option_type"].lower() == "binary":
        params["threshold"] = config["threshold"]
        params["binary_payout"] = config["binary_payout"]
    
    return params

# Analytical Solution for European Option Price (Black-Scholes)
def black_scholes(S_0, K, T, r, q, sigma, option_type = 'call'):
    '''
    Analytical Solution for the Price of European Options.
    Takes as inputs, the current price of the underlying, the strike price of the option, expiration, 
    interest rate r, implied volatility (sigma), and the option type (call/put), and returns the "fair"
    option price. 
    '''

    option_type = option_type.lower()
    if T <= 0 or sigma <= 0:
        # intrinsic value discounted appropriately (degenerate cases)
        disc_r = np.exp(-r * max(T, 0.0))
        disc_q = np.exp(-q * max(T, 0.0))
        if option_type == 'call':
            return max(S_0*disc_q - K*disc_r, 0.0)
        elif option_type == 'put':
            return max(K*disc_r - S_0*disc_q, 0.0)
        else:
            raise ValueError('Option type must be "call" or "put"')

    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_type == 'call':
        return S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == 'put':
        return K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)
    else:
        raise ValueError('Option type must be "call" or "put"')

# Monte-Carlo Simulation
def timestep(params):
    '''
    Simulates one timestep in the Monte-Carlo simulation. Takes a given start_price at the beginning of the timestep 
    and transforms it into a random price (acc. to GBM) after the timestep.
    '''
    detla_t=params["expiration"]/params["nr_of_timesteps"]
    return params["s_0"] * np.exp((params["r"]-params["q"]-0.5*params["iv"]**2)*delta_t + params["sigma"] * np.sqrt(delta_t)*np.random.normal(0,1))


def simulation_run(params):
    '''
    One full pricing simulation for n timesteps. Takes a given start_price as well as the number of timesteps, and simulates
    an asset's random price development (acc. to GBM) over the n timesteps. Returns an array of the format: [[timestep, price]]
    '''
    prices = [[0, params["s_0"]]]
    S_i = params["s_0"]
    for i in range(params["nr_of_timesteps"]):
        S_i = timestep(params)
        prices.append([i+1, S_i])
    return prices


def MonteCarlo(params):
    '''
    Monte-Carlo simulation for an Asset's price development, given a start_price, the assets volatility, current interest rate, dividend yield, expiration
    the number of timesteps per simulation and the total number of simulations. Simulates M different price developments with n timesteps each. 
    Each price development follows a Stochastic Wiener process (i.e. Geometric Brownian Motion).
    '''
    list_of_prices = []
    for _ in range(params["nr_of_simulations"]):
        list_of_prices.append(simulation_run(params))
    return list_of_prices


# Testrun
# print(MonteCarlo(10, 10, 100, 1/12, 0.04, 0.01, 0.30))

def mc_pricing_basic(option_function, params):
    paths=MonteCarlo(params)
    payoffs = [option_function(path, params) for path in paths]

    mean_payoff = np.mean(payoffs)
    std_error = np.std(payoffs, ddof=1) / math.sqrt(len(payoffs))
    price = math.exp(-params["r"] * params["expiration"]) * mean_payoff
    return price, std_error

def payoff_european(path, params):
    ST = path[-1][1]
    strike = params["k_0"]
    if params["option_type"] == "call": 
        return max(ST - strike, 0.0)
    else:                       
        return max(strike - ST, 0.0)

def payoff_binary(path, params):
    ST = path[-1][1]
    K  = params["threshold"]
    Q  = float(params.get("binary_payout", 1.0))

    if params["option_type"] == "call":
        return Q if ST > K else 0.0
    else:
        return Q if ST < K else 0.0

def payoff_asian_arith(path, params):
    strike = params["k_0"]
    prices = [S for _, S in path]
    avg_price = sum(prices) / len(prices)

    if params["option_type"] == "call":
        return max(avg_price - strike, 0.0)
    else:
        return max(strike - avg_price, 0.0)
# Testrun
filename_csv = os.path.join(os.getcwd(),'try.csv')
config = read_input_file(filename_csv)
# print(calculate_expiration("2025-10-25", "09:30", "2026-01-25", "PM"))
params = transform_input(filename_csv)
print(mc_pricing_basic(option_function=payoff_european, params=params))
