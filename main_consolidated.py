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
      
    params["expiration"] = calculate_expiration(config["start_date"], config["start_time"], config["expiration_date"], config["expiration_time"])

    try:
        params["s_0"] = float(config["underlying_price"])
    except (ValueError, TypeError):
        raise ValueError("Underlying price has to be of type 'float'!")

    try: 
        params["k_0"] = float(config["option_strike"])
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



