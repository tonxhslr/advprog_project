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


def transform_input(file):
    '''
    Transforms all the inputs from a file into actual usable variables for the model

    Returns: Dictionary of inputs
    '''

    def calculate_expiration(start_date, start_time, expiration_date, expiration_time):
        '''
        Calculates time until expiration as a fraction of a year, given the start/end dates and times. 
        '''
    
        try:
            start = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            raise ValueError("Invalid or missing date/time. Expected YYYY-MM-DD and HH:MM.")
          
        if expiration_time == "AM":
            end_time = "09:00"
        elif expiration_time == "PM":
            end_time = "16:00"
        else:
            raise ValueError("Expiration must be either AM or PM")
        try: 
            end = datetime.strptime(f"{expiration_date} {end_time}", "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            raise ValueError("Invalid or missing date/time. Expected YYYY-MM-DD and HH:MM.")
    
        return ((end-start).total_seconds())/(365.25*24*60*60)

    params = {}

    config = read_input_file(file)
    
    params["option_type"] = config["option_type"].lower()
    if params["option_type"] not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'!")
      
    params["exercise_type"] = config["exercise_type"].lower()
    if params["exercise_type"] not in ("european", "american", "asian", "barrier", "binary"):
        raise ValueError("Exercise type must be 'european', 'american', 'asian', 'barrier' or 'binary'!")
      
    params["T"] = calculate_expiration(config["start_date"], config["start_time"], config["expiration_date"], config["expiration_time"])

    try:
        params["S_0"] = float(config["underlying_price"])
    except (ValueError, TypeError):
        raise ValueError("Underlying price has to be of type 'float'!")

    try: 
        params["K"] = float(config["option_strike"])
    except (ValueError, TypeError):
        raise ValueError("Strike price has to be of type 'float'!")

    try:
        params["iv"] = float(config["volatility"])/100
    except (ValueError, TypeError): 
        raise ValueError("Volatility has to be of type 'float' (x.x%)!")

    try:
        params["r"] = float(config["interest_rate"])/100
    except (ValueError, TypeError):
        raise ValueError("Interest rate has to be of type 'float' (x.x%)!")
    
    if config["dividend"] == "None" or config["dividend"] is False:
        params["q"] = 0
    else: 
        try: 
            params["q"] = float(config["dividend"])
        except (ValueError, TypeError):
            raise ValueError("Dividend has to be of type 'float'!")

    try: 
        params["nr_of_simulations"] = int(config["nr_simulations"])
    except (ValueError, TypeError):
        raise ValueError("Number of simulations has to be of type 'int'!")

    try: 
        params["nr_of_timesteps"] = int(config["nr_of_timesteps"])
    except (ValueError, TypeError):
        raise ValueError("Number of timesteps has to be of type 'int'!")
    
    if bool(config["output_to_file"]) == False: 
        params["filename"] = None
    elif bool(config["output_to_file"]) == True: 
        params["filename"] = config["output_filename"]
        if Path(params["filename"]).suffix.lower() not in (".csv", ".json"):
            raise ValueError("File has to be of '.csv' or '.json'-format. Please add the according suffix!")
    else: 
        raise ValueError("Please enter an option for Output To File")

    # Currently, the program will overwrite files, but one could implement a check for existing files and prompt the user, whether he wants to overwrite the file
    
    if config["option_type"].lower() == "barrier":
        params["barrier_type"] = config["barrier_type"]
        if not any(word in params["barrier_type"].lower() for word in ("in", "out")):
            raise ValueError("Barrier type must contain 'in' or 'out' (e.g. 'knock-in', 'knockout')!")   
        try: 
            params["threshold"] = float(config["threshold"])
        except (ValueError, TypeError): 
            raise ValueError("Barrier threshold has to be type 'float'!")

    if config["option_type"].lower() == "binary":
        try: 
            params["threshold"] = float(config["threshold"])
        except (ValueError, TypeError): 
            raise ValueError("Binary threshold has to be type 'float'!")
        try:
            params["binary_payout"] = float(config["binary_payout"])
        except (ValueError, TypeError):
            raise ValueError("Binary payout has to be type 'float'!")
    
    return params


# 2) Black-Scholes: Analytical Solution for European or Binary Option Price, or American options without Dividends (Black-Scholes)
def black_scholes(params):
    """
    Analytical Solution for:
      - European call/put (with dividend yield q)
      - Binary cash-or-nothing call/put
    Accepts 'american' in exercise_type, but returns the European value because
    American options with dividends do not have a closed-form Black–Scholes price.
    """
    
    S_0 = params["S_0"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    q = params["q"]
    sigma = params["iv"]
    option_type = params.get("option_type", 'call')
    exercise_type = params.get("exercise_type", 'european')
    payout = params.get("binary_payout", 1.0)

    if T <= 0 or sigma <= 0:
        disc_r = np.exp(-r * max(T, 0.0))
        disc_q = np.exp(-q * max(T, 0.0))
        itm = (S_0 >= K) if option_type == 'call' else (S_0 <= K)

        if exercise_type == 'binary':
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

    if exercise_type == 'european':
        if option_type == 'call':
            return S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)

    elif exercise_type == 'binary':
        if option_type == 'call':
            return payout * disc_r * norm.cdf(d2)
        elif option_type == 'put':
            return payout * disc_r * norm.cdf(-d2)

    elif exercise_type == 'american':
        # price it as european here; true american (esp. with dividends)
        # should be done with a binomial / LSM outside this function
        if option_type == 'call':
            euro_price = S_0 * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
            return euro_price
        else:  # put
            euro_price = K * disc_r * norm.cdf(-d2) - S_0 * disc_q * norm.cdf(-d1)
            return euro_price

    else:
        raise ValueError("exercise_type must be 'european', 'binary', or 'american'")



#3) Monte-Carlo Simulation

'''
The functions below take the full 'params' dictionary as input. This design keeps a consistent interface across all pricing and payoff methods,
making it easier to combine, extend, and reuse different pricing approaches without changing function signatures.
'''


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
                Volatility (sigma): params["iv"]
            '''
            T = params["T"]
            nr_of_timesteps = params["nr_of_timesteps"]
            S_0 = params["S_0"]
            r = params["r"]
            q = params["q"]
            sigma = params["iv"]
            
            delta_t=exp/nr_time
            return params["s_0"] * np.exp((r-q-0.5*iv**2)*delta_t + iv * np.sqrt(delta_t)*np.random.normal(0,1))

# Simulation Run -----------------------------------------------------------------------------------------------------------------------

        nr_time = params["nr_of_timesteps"]
        start_price = params["s_0"]
        prices = [[0, start_price]]
        S_i = start_price
        for i in range(nr_time):
            S_i = timestep(params)
            prices.append([i+1, S_i])
        return prices

# Monte-Carlo -------------------------------------------------------------------------------------------------------------------------

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
        Option type option_type: params["option_type"]
    '''

    K = params["K"]
    option_type = params["option_type"]

    ST = path[-1][1]
    
    if option_type == "call": 
        return max(ST - K, 0.0)
    else:                       
        return max(K - ST, 0.0)

def payoff_binary(path, params):
    '''
    The binary payoff function that calulates the payoff taking the last price of the path and comparing it against a threshold, 
    returning a binary payout if the threshold is exceeded and 0 if not.

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Option type option_type: params["option_type"]
        Threshold K: params["threshold"]
        Binary payout Q: params["binary_payout"]
    '''
    threshold  = params["threshold"]
    payout = params.get("binary_payout", 1.0)
    option_type = params["option_type"]

    ST = path[-1][1]

    if option_type == "call":
        return payout if ST > threshold else 0.0
    else:
        return payout if ST < threshold else 0.0

def payoff_asian(path, params):
    '''
    The asian payoff function that calulates the payoff taking the average price of the path 
    and uses it to calculate the payoff against strike

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike: params["k_0"]
        Option type option_type: params["option_type"]
    '''
    K = params["K"]
    option_type = params["option_type"]
    prices = [S[1] for _, S in path]
    avg_price = sum(prices) / len(prices)

    if option_type == "call":
        return max(avg_price - K, 0.0)
    else:
        return max(K - avg_price, 0.0)

def payoff_barrier(path, params):
    '''
    Barrier option payoff that uses the simulated path to determine whether the barrier was hit (touch-inclusive),
    then returns the regular, european payoff at expiry using the path’s last price if the knock condition is satisfied. 

    Inputs:
    - path:  list of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike: params["K"]
        Start price start_price: params["s_0"]
        Option type option_type: params["option_type"]
        Threshold K: params["threshold"]
        Barrier type b_type: params["barrier_type"]
    '''
    
    K = params["K"]
    threshold = params["threshold"]
    start_price = params["S_0"]
    option_type = params["option_type"]
    barrier_type = params["barrier_type"]

    ST = path[-1][1]
    
    european_payoff = max(ST - K, 0.0) if option_type == "call" else max(K - ST, 0.0)

    if threshold == start_price:
        hit = True
    else:
        up = threshold > start_price
        if up:
            hit = any(S[1] >= threshold for _, S in path)
        else:
            hit = any(S[1] <= threshold for _, S in path)

    if "in" in barrier_type:
        return european_payoff if hit else 0.0
    elif "out" in barrier_type:
        return 0.0 if hit else european_payoff

def mc_pricing_basic(params):
"""
Monte Carlo pricing function.

Generates simulated price paths, computes payoffs using the appropriate payoff function, and returns the discounted mean 
as the theoretical option price. Uses the full 'params' dictionary for a consistent interface across pricing methods.

Inputs:
 - params: dictionary of parameters, from which the function takes
     Excercise type (european, asian, etc.) exercise_type: params["exercise_type"]
     Interest rate r: params["r"]
     Expiration T: params["expiration"]

     + all the parameters used to calculate paths in MonteCarlo function

"""
    exercise_type = params["exercise_type"]
    r = params["r"]
    T = params["T"]

    paths=MonteCarlo(params)

    payoff_functions = {
        "european" : payoff_european,
        "binary" : payoff_binary,
        "asian" : payoff_asian,
        "barrier" : payoff_barrier}
    
    option_function=payoff_functions[exercise_type]
    payoffs = [option_function(path, params) for path in paths]

    mean_payoff = np.mean(payoffs)
    price = math.exp(-r * T) * mean_payoff
    return price


# 4) Greeks

def bs_greeks(params):
  
    option_type = params["option_type"]
    S_0 = params["S_0"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    q = params["q"]
    sigma = params["iv"]
    
    if T <= 0 or sigma <= 0 or S_0 <= 0 or K <= 0:
        return np.nan, np.nan

    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    
    vega = s_0 * np.exp(-q * T) * nd1 * np.sqrt(T)
    
    term1 = -(s_0 * np.exp(-q * T) * nd1 * sigma) / (2 * np.sqrt(T))
    
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    
    if option_type == "call":
        delta = disc_q * Nd1
        rho = K * T * disc_r * Nd2
        term2 = q * s_0 * np.exp(-q * T) * Nd1
        term3 = r * K * np.exp(-r * T) * Nd2
        theta = term1 - term2 - term3
    else:
        delta = disc_q * (Nd1 - 1.0)
        rho = -K * T * disc_r * norm.cdf(-d2)
        term2 = q * s_0 * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = term1 + term2 + term3
    
    gamma = disc_q * nd1 / (S_0 * sigma * np.sqrt(T))
    
    return float(delta), float(gamma), float(rho), float(theta), float(vega)


def option_calculator(file):
    '''
    Final Option Calculator Function

    Takes the Input-File and calculates option price both analytically (if possible) as well as numerically, using Monte-Carlo Simulation, and all Greeks.
    If specified, it writes the calculated values into a file of either '.csv' or '.json' format. 
    If it shouldn't write the output into a file, it will print the output into the terminal. 
    
    Returns: Dictionary with all calculated values
    '''
    
    params = transform_input(file)
    option_type = params["option_type"]
    output_file = params["filename"]

    option_price_analytical = 0
    option_price_mc = 0
    
    if option_type == "european" or option_type == "binary" or (option_type == "american" and params["q"] == 0):
        option_price_analytical = black_scholes(params)

    elif option_type == "asian" or option_type == "barrier" or option_type == "european" or option_type == "binary":
        option_price_mc = mc_pricing_basic(params)

    elif (option_type =="american" and params["q"] != 0): 
        option_price_mc = mc_pricing_american(params) 

    delta, gamma, rho, theta, vega = bs_greeks(params)

    output = {
        "Option Price (Analytical (BS))": option_price_analytical,
        "Option Price (Numerical (MC))": option_price_mc,
        "Delta": delta,
        "Gamma": gamma,
        "Rho": rho, 
        "Theta": theta,
        "Vega": vega
    }

    if output_file:
        output_file = Path(output_file)
        suffix = output_file.suffix.lower()  # ".csv" or ".json"
        
        if suffix == ".csv":
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(output.keys())
                writer.writerow(output.values())
        elif suffix == ".json":
            with open(output_file, "w") as f:
                json.dump(output, f, indent=4)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
    else: 
        print("Option Price (Analytical (BS)):", option_price_analytical)
        print("Option Price (Numerical (MC)):", option_price_mc)
        print("Delta:", delta)
        print("Gamma:", gamma)
        print("Rho:", rho)
        print("Theta:", theta)
        print("Vega:", vega)

    return output
        
        
        


