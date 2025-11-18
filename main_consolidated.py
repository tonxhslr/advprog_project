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
