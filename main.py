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
    params["nr_of_simulations"] = int(config["nr_simulations"])
    params["nr_of_timesteps"] = int(config["time_step"])
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

# Analytical Solution for European, Binary Option Price (Black-Scholes)
def black_scholes(S_0, K, T, r, q, sigma, option_type='call', option_style='european', payout=1.0):
    '''
    Analytical Solution for the Price of European Options.
    Takes as inputs, the current price of the underlying, the strike price of the option, expiration, 
    interest rate r, implied volatility (sigma), and the option type (call/put), and returns the "fair"
    option price. 
    '''

    option_type = option_type.lower()
    option_style = option_style.lower()

    if T <= 0 or sigma <= 0:
        disc_r = np.exp(-r * max(T, 0.0))
        disc_q = np.exp(-q * max(T, 0.0))
        itm = (S_0 >= K) if option_type == 'call' else (S_0 <= K)

        if option_style == 'binary':
            return payout * disc_r if itm else 0.0
        else:  # european
            if option_type == 'call':
                return max(S_0 * disc_q - K * disc_r, 0.0)
            else:
                return max(K * disc_r - S_0 * disc_q, 0.0)

    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_style == 'european':
        if option_type == 'call':
            return S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)

    elif option_style == 'binary':
        if option_type == 'call':
            return payout * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return payout * disc_r * norm.cdf(-d2)

    else:
        raise ValueError("option_style must be 'european' or 'binary'")
        
# Analytical Solution for European, Binary Option Price (Black-Scholes) (INCLUDE AMERICAN OPTIONS WITH DIVIDEND YIELDS)
def black_scholes(S_0, K, T, r, q, sigma, option_type='call', option_style='european', payout=1.0):
    """
    Analytical Solution for:
      - European call/put (with dividend yield q)
      - Binary cash-or-nothing call/put
    Accepts 'american' in option_style, but returns the European value because
    American options with dividends do not have a closed-form Black–Scholes price.
    """
    
    option_type = option_type.lower()
    option_style = option_style.lower()

    if T <= 0 or sigma <= 0:
        disc_r = np.exp(-r * max(T, 0.0))
        disc_q = np.exp(-q * max(T, 0.0))
        itm = (S_0 >= K) if option_type == 'call' else (S_0 <= K)

        if option_style == 'binary':
            return payout * disc_r if itm else 0.0
        else:  # european / american fallback
            if option_type == 'call':
                return max(S_0 * disc_q - K * disc_r, 0.0)
            else:
                return max(K * disc_r - S_0 * disc_q, 0.0)

    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_style == 'european':
        if option_type == 'call':
            return S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)

    elif option_style == 'binary':
        if option_type == 'call':
            return payout * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return payout * disc_r * norm.cdf(-d2)

    elif option_style == 'american':
        # price it as european here; true american (esp. with dividends)
        # should be done with a binomial / LSM outside this function
        if option_type == 'call':
            euro_price = S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
            return euro_price
        else:  # put
            euro_price = K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)
            return euro_price

    else:
        raise ValueError("option_style must be 'european', 'binary', or 'american'")

# Monte-Carlo Simulation

'''
The functions below take the full 'params' dictionary as input. This design keeps a consistent interface across all pricing and payoff methods,
making it easier to combine, extend, and reuse different pricing approaches without changing function signatures.
'''

def timestep(params):
    '''
    Simulates one timestep in the Monte-Carlo simulation. Takes a given start_price at the beginning of the timestep 
    and transforms it into a random price (acc. to GBM) after the timestep.
    
    Inputs:
     - params: dictionary of parameters, from which the function takes
        Expiration exp: params["expiration"]
        Number of timesteps nr_time: params["nr_of_timesteps"]
        Start price start_price: params["s_0"]
        Interest rate r: params["r"]
        Dividend q: params["q"]
        Volatility iv: params["iv"]
    '''
    exp = params["expiration"]
    nr_time = params["nr_of_timesteps"]
    start_price = params["s_0"]
    r = params["r"]
    q = params["q"]
    iv = params["iv"]
    
    delta_t=exp/nr_time
    return params["s_0"] * np.exp((r-q-0.5*iv**2)*delta_t + iv * np.sqrt(delta_t)*np.random.normal(0,1))


def simulation_run(params):
    '''
    One full pricing simulation for n timesteps. Takes a given start_price as well as the number of timesteps, and simulates
    an asset's random price development (acc. to GBM) over the n timesteps. Returns an array of the format: [[timestep, price]]

    Inputs:
     - params: dictionary of parameters, from which the function takes
        Number of timesteps nr_time: params["nr_of_timesteps"]
        Start price start_price: params["s_0"]

        + parameters for timestep function

    '''
    nr_time = params["nr_of_timesteps"]
    start_price = params["s_0"]
    prices = [[0, start_price]]
    S_i = start_price
    for i in range(nr_time):
        S_i = timestep(params)
        prices.append([i+1, S_i])
    return prices


def MonteCarlo(params):
    '''
    Monte-Carlo simulation for an Asset's price development, given a start_price, the assets volatility, current interest rate, dividend yield, expiration
    the number of timesteps per simulation and the total number of simulations. Simulates M different price developments with n timesteps each. 
    Each price development follows a Stochastic Wiener process (i.e. Geometric Brownian Motion).

    Inputs:
     - params: dictionary of parameters, from which the function takes
        Number of simulations nr_sim: params["nr_of_simulations"]
    
        + parameters used for simulation_run function
    '''
    nr_sim = params["nr_of_simulations"]
    
    list_of_prices = []
    for _ in range(params["nr_of_simulations"]):
        list_of_prices.append(simulation_run(params))
    return list_of_prices

"""
The payoff functions return a payoff given a single simulated price path.
They all take two arguments:
  - path: a list of [timestep, price] pairs from a Monte Carlo simulation
  - params: a dictionary containing option and model parameters
Each function implements a specific option type (e.g., European, Asian, Binary, Barrier).
"""

def payoff_european(path, params):
    '''
    The european payoff function that calulates the payoff taking the last price of the path,
    and uses it to calculate the payoff against strike.

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike: params["k_0"]
        Option type opt_type: params["option_type"]
    '''

    strike = params["k_0"]
    opt_type = params["option_type"]

    ST = path[-1][1]
    
    if opt_type == "call": 
        return max(ST - strike, 0.0)
    else:                       
        return max(strike - ST, 0.0)

def payoff_binary(path, params):
    '''
    The binary payoff function that calulates the payoff taking the last price of the path and comparing it against a threshold, 
    returning a binary payout if the threshold is exceeded and 0 if not.

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Option type opt_type: params["option_type"]
        Threshold K: params["threshold"]
        Binary payout Q: params["binary_payout"]
    '''
    K  = params["threshold"]
    Q  = params.get("binary_payout", 1.0)
    opt_type = params["option_type"]

    ST = path[-1][1]

    if opt_type == "call":
        return Q if ST > K else 0.0
    else:
        return Q if ST < K else 0.0

def payoff_asian(path, params):
    '''
    The asian payoff function that calulates the payoff taking the average price of the path 
    and uses it to calculate the payoff against strike

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike: params["k_0"]
        Option type opt_type: params["option_type"]
    '''
    strike = params["k_0"]
    opt_type = params["option_type"]
    prices = [S[1] for _, S in path]
    avg_price = sum(prices) / len(prices)

    if opt_type == "call":
        return max(avg_price - strike, 0.0)
    else:
        return max(strike - avg_price, 0.0)

def payoff_barrier(path, params):
    '''
    Barrier option payoff that uses the simulated path to determine whether the barrier was hit (touch-inclusive),
    then returns the regular, european payoff at expiry using the path’s last price if the knock condition is satisfied. 

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike: params["k_0"]
        Start price start_price: params["s_0"]
        Option type opt_type: params["option_type"]
        Threshold K: params["threshold"]
        Barrier type b_type: params["barrier_type"]
    '''
    
    strike = params["k_0"]
    K  = params["threshold"]
    start_price = params["s_0"]
    opt_type = params["option_type"]
    b_type = params["barrier_type"]

    ST = path[-1][1]
    
    european_payoff = max(ST - strike, 0.0) if opt_type == "call" else max(strike - ST, 0.0)

    if K == start_price:
        hit = True
    else:
        up = K > start_price
        if up:
            hit = any(S[1] >= K for _, S in path)
        else:
            hit = any(S[1] <= K for _, S in path)

    if "in" in b_type:
        return european_payoff if hit else 0.0
    elif "out" in b_type:
        return 0.0 if hit else european_payoff

def mc_pricing_basic(params):
"""
Monte Carlo pricing function.

Generates simulated price paths, computes payoffs using the appropriate payoff function, and returns the discounted mean 
as the theoretical option price. Uses the full 'params' dictionary for a consistent interface across pricing methods.

Inputs:
 - params: dictionary of parameters, from which the function takes
     Excercise type (european, asian, etc.) exc_type: params["exercise_type"]
     Interest rate r: params["r"]
     Expiration exp: params["expiration"]

     + all the parameters used to calculate paths in MonteCarlo function

"""
    exc_type = params["exercise_type"]
    r = params["r"]
    exp = params["expiration"]

    paths=MonteCarlo(params)

    payoff_functions = {
        "european" : payoff_european,
        "binary" : payoff_binary,
        "asian" : payoff_asian,
        "barrier" : payoff_barrier}
    
    option_function=payoff_functions[exc_type]
    payoffs = [option_function(path, params) for path in paths]

    mean_payoff = np.mean(payoffs)
    price = math.exp(-r * exp) * mean_payoff
    return price


# Testrun
filename_csv = os.path.join(os.getcwd(),'try.csv')
config = read_input_file(filename_csv)
# print(calculate_expiration("2025-10-25", "09:30", "2026-01-25", "PM"))
params = transform_input(filename_csv)
print(mc_pricing_basic(params))
