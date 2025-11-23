import numpy as np
import math
from pathlib import Path
from scipy.stats import norm
import json
import csv
import os
from datetime import date, datetime

np.random.seed(42)  # For reproducibility during testing

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
        if not start_date or str(start_date) == "":
            start_date = date.today().strftime("%Y-%m-%d")
        if not start_time or str(start_time) == "":
            start_time = date.today().strftime("%H:%M")
        
        try:
            start = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            raise ValueError("Invalid or missing date/time. Expected YYYY-MM-DD and HH:MM.")
          
        if expiration_time == "AM" or expiration_time =="":
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
    
    params["option_type"] = (config["option_type"] or "call").lower()
    if params["option_type"] not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'!")
      
    params["exercise_type"] = (config["exercise_type"] or "european").lower()
    if params["exercise_type"] not in ("european", "american", "asian", "barrier", "binary"):
        raise ValueError("Exercise type must be 'european', 'american', 'asian', 'barrier' or 'binary'!")
      
    params["T"] = calculate_expiration(config["start_date"], config["start_time"], config["expiration_date"], config["expiration_time"])

    try:
        params["S_0"] = float(config["underlying_price"] or 100)
    except (ValueError, TypeError):
        raise ValueError("Underlying price has to be of type 'float'!")

    try: 
        params["K"] = float(config["option_strike"] or 100)
    except (ValueError, TypeError):
        raise ValueError("Strike price has to be of type 'float'!")

    try:
        params["iv"] = float(config["volatility"] or 20)/100
    except (ValueError, TypeError): 
        raise ValueError("Volatility has to be of type 'float' (x.x%)!")

    try:
        params["r"] = float(config["interest_rate"] or 1.5)/100
    except (ValueError, TypeError):
        raise ValueError("Interest rate has to be of type 'float' (x.x%)!")
    
    try: 
        params["q"] = float(config["dividend"] or 0)
    except (ValueError, TypeError):
        raise ValueError("Dividend has to be of type 'float'!")
    params["q"] = params["q"] / params["S_0"]       # Transforming cash-dividend into %-yield
    
    try: 
        params["nr_of_simulations"] = int(config["nr_of_simulations"] or 1000)
    except (ValueError, TypeError):
        raise ValueError("Number of simulations has to be of type 'int'!")

    try: 
        params["nr_of_timesteps"] = int(config["nr_of_timesteps"] or 100)
    except (ValueError, TypeError):
        raise ValueError("Number of timesteps has to be of type 'int'!")
    
    if str(config["output_to_file"]).strip().lower() in ("false", "0", "none", ""):
        params["filename"] = None

    elif str(config["output_to_file"]).strip().lower() in ("true", "1", "yes"):
        params["filename"] = config["output_filename"]
        if Path(params["filename"]).suffix.lower() not in (".csv", ".json"):
            raise ValueError("File has to be of '.csv' or '.json'-format. Please add the according suffix!")

    else:
        raise ValueError("Please enter an option for Output To File")

    # Currently, the program will overwrite files, but one could implement a check for existing files and prompt the user, whether he wants to overwrite the file
    
    if config["exercise_type"].lower() == "barrier":
        params["barrier_type"] = (config["barrier_type"] or "knockin")
        if not any(word in params["barrier_type"].lower() for word in ("in", "out")):
            raise ValueError("Barrier type must contain 'in' or 'out' (e.g. 'knock-in', 'knockout')!")   
        try: 
            params["threshold"] = float(config["threshold"] or 100)
        except (ValueError, TypeError): 
            raise ValueError("Barrier threshold has to be type 'float'!")

    if config["option_type"].lower() == "binary":
        try: 
            params["threshold"] = float(config["threshold"] or 100)
        except (ValueError, TypeError): 
            raise ValueError("Binary threshold has to be type 'float'!")
        try:
            params["binary_payout"] = float(config["binary_payout"] or 1.0)
        except (ValueError, TypeError):
            raise ValueError("Binary payout has to be type 'float'!")
    
    return params


# 2) Black-Scholes: Analytical Solution for European or Binary Option Price, or American options without Dividends (Black-Scholes)
def black_scholes(params: dict) -> float:
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
    payout = params.get("binary_payout", None)

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


def monte_carlo(params: dict) -> np.array:
    '''
    Monte-Carlo simulation for an Asset's price development, given a start_price, the assets volatility, current interest rate, dividend yield, expiration
    the number of timesteps per simulation and the total number of simulations. Simulates M different price developments with n timesteps each. 
    Each price development follows a Stochastic Wiener process (i.e. Geometric Brownian Motion).

    Inputs:
     - params: dictionary of parameters, from which the function takes
        Number of simulations (nr_of_simulations): params["nr_of_simulations"]
    
        + parameters used for simulation_run function
    '''

    def simulation_run(params: dict) -> list:
        '''
        One full pricing simulation for n timesteps. Takes a given start_price as well as the number of timesteps, and simulates
        an asset's random price development (acc. to GBM) over the n timesteps. Returns an array of the format: [[timestep, price]]
    
        Inputs:
         - params: dictionary of parameters, from which the function takes
            Number of timesteps nr_of_timesteps: params["nr_of_timesteps"]
            Start price (S_0): params["S_0"]
    
            + parameters for timestep function

        '''

        def timestep(start_price: float, params: dict) -> float:
            '''
             Simulates one timestep in the Monte-Carlo simulation. Takes a given start_price at the beginning of the timestep 
            and transforms it into a random price (acc. to GBM) after the timestep.
            
            Inputs:
            - S_prev: previous price
             - params: dictionary of parameters, from which the function takes
                Expiration (T): params["T"]
                Number of timesteps (nr_of_timesteps): params["nr_of_timesteps"]
                Start price (S_0): params["S_0"]
                Interest rate (r): params["r"]
                Dividend (q): params["q"]
                Volatility (sigma): params["iv"]
            '''
            T = params["T"]
            nr_of_timesteps = params["nr_of_timesteps"]
            r = params["r"]
            q = params["q"]
            sigma = params["iv"]
            
            delta_t=T/nr_of_timesteps

            return start_price * np.exp((r-q-0.5*sigma**2)*delta_t + sigma * np.sqrt(delta_t)*np.random.normal(0,1)) 

# Simulation Run -----------------------------------------------------------------------------------------------------------------------

        nr_of_timesteps = params["nr_of_timesteps"]
        S_0 = params["S_0"]
        prices = [[0, S_0]]
        S_i = S_0
        for i in range(nr_of_timesteps):
            S_i = timestep(S_i, params)
            prices.append([i+1, S_i])
        return prices

# Monte-Carlo -------------------------------------------------------------------------------------------------------------------------

    nr_of_simulations = params["nr_of_simulations"]
    
    list_of_prices = []
    for _ in range(nr_of_simulations):
        list_of_prices.append(simulation_run(params))
    return np.array(list_of_prices)


"""
The payoff functions return a payoff given a single simulated price path.
They all take two arguments:
  - path: a numpy array of [timestep, price] pairs from a Monte Carlo simulation
  - params: a dictionary containing option and model parameters
Each function implements a specific option type (e.g., European, Asian, Binary, Barrier).
"""

def payoff_european(paths: np.array, params: dict) -> np.array:
    '''
     The european payoff function that calulates the payoff taking the last price of the path,
    and uses it to calculate the payoff against strike.

    Inputs:
    - path:  np.array of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike (K): params["K"]
        Option type (option_type): params["option_type"]
    '''

    K = params["K"]
    option_type = params["option_type"]

    S_T = paths[:, -1, 1]
    
    if option_type == "call": 
        return np.maximum(S_T - K, 0.0)
    else:                       
        return np.maximum(K - S_T, 0.0)

def payoff_binary(paths: np.array, params: dict) -> np.array:
    '''
     The binary payoff function that calulates the payoff taking the last price of the path and comparing it against a threshold, 
    returning a binary payout if the threshold is exceeded and 0 if not.

    Inputs:
    - path:  np.array of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Threshold (threshold): params["threshold"]
        Binary payout (payout): params["binary_payout"]
        Option type (option_type): params["option_type"]
    '''
    threshold  = params["threshold"]
    payout = params["binary_payout"]
    option_type = params["option_type"]

    S_T = paths[:, -1, 1]

    if option_type == "call":
        return np.where(S_T > threshold, payout, 0.0)
    else:
        return np.where(S_T < threshold, payout, 0.0)

def payoff_asian(paths: np.array, params: dict) -> np.array:
    '''
    The asian payoff function that calulates the payoff taking the average price of the path 
    and uses it to calculate the payoff against strike

    Inputs:
    - path:  np.array of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike (K): params["K"]
        Option type (option_type): params["option_type"]
    '''
    K = params["K"]
    option_type = params["option_type"]
    prices = paths[:,:,1]
    avg_price = np.average(prices, axis=1)

    if option_type == "call":
        return np.maximum(avg_price - K, 0.0)
    else:
        return np.maximum(K - avg_price, 0.0)

def payoff_barrier(paths: np.array, params: dict) -> np.array:
    '''
    Barrier option payoff that uses the simulated path to determine whether the barrier was hit (touch-inclusive),
    then returns the regular, european payoff at expiry using the path’s last price if the knock condition is satisfied. 

    Inputs:
    - path:  np.array of timestep,price pairs generated by Monte Carlo function
    - params: dictionary of parameters, from which the function takes
        Strike (K): params["K"]
        Start price (S_0): params["S_0"]
        Threshold (threshold): params["threshold"]
        Option type (option_type): params["option_type"]
        Barrier type (barrier_type): params["barrier_type"]
    '''
    
    K = params["K"]
    threshold = params["threshold"]
    S_0 = params["S_0"]
    option_type = params["option_type"]
    barrier_type = params["barrier_type"]

    prices = paths[:,:,1]
    S_T = prices[:,-1]
    
    european_payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)

    if threshold == S_0:
        hit = np.ones(prices.shape[0], dtype=bool)
    else:
        up = threshold > S_0
        if up:
            hit = np.any(prices >= threshold, axis=1)
        else:
            hit = np.any(prices <= threshold, axis=1)

    if "in" in barrier_type:
        payoff = np.where(hit, european_payoff, 0.0)
    elif "out" in barrier_type:
        payoff = np.where(hit, 0.0, european_payoff)
    
    return payoff

def mc_pricing_basic(params: dict) -> float:
    """
     Monte Carlo pricing function.
    
    Generates simulated price paths, computes payoffs using the appropriate payoff function, and returns the discounted mean 
    as the theoretical option price. Uses the full 'params' dictionary for a consistent interface across pricing methods.
    
    Inputs:
     - params: dictionary of parameters, from which the function takes
         Excercise type (european, asian, etc.) (exercise_type): params["exercise_type"]
         Interest rate (r): params["r"]
         Expiration (T): params["T"]
    
         + all the parameters used to calculate paths in MonteCarlo function

    """
    exercise_type = params["exercise_type"]
    r = params["r"]
    T = params["T"]

    paths=monte_carlo(params)

    payoff_functions = {
        "european" : payoff_european,
        "binary" : payoff_binary,
        "asian" : payoff_asian,
        "barrier" : payoff_barrier}
    
    option_function=payoff_functions[exercise_type]
    payoffs = option_function(paths, params)

    mean_payoff = np.mean(payoffs)
    price = math.exp(-r * T) * mean_payoff
    return price


def longstaff_schwartz_pricing_american(params: dict) -> float:
    '''
    Longstaff-Schwartz Pricing for American Options (Early Exercise). 

    Takes the array of simulated price paths from the Monte-Carlo simulation and uses backward induction, starting from the last period, 
    in each period comparing the immediate exercise value against the expected continuation value (which is estimated using linear regression).
    If immediate exercise value exceeds continuation value, payoff is discounted and added to the list of payoffs. 
    '''
    
    def payoff_american_matrix(paths: np.array, params: dict) -> np.array:
        '''
        Regular Option-Payoff Function, but over the entire matrix of price developments, instead of only final values as for European Options.
        Returns an array with the same dimensions as the array of price-developments, but with immediate payoff values in each timestep.
        '''
        
        option_type = params["option_type"]
        K = params["K"]

        if option_type == "call":
            payoffs = np.maximum(paths[:,:,1] - K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(K - paths[:,:,1], 0.0)
        else:
            raise ValueError("Option Type has to be 'call' or 'put'")
        
        return payoffs

# Longstaff-Schwartz -----------------------------------------------------------------------------------

    r = params["r"]
    T = params["T"]
    nr_of_timesteps = params["nr_of_timesteps"]
    discount_step = np.exp(-r * T/nr_of_timesteps)

    paths = monte_carlo(params)
    payoffs = payoff_american_matrix(paths, params)

    # Cashflow at Maturity, t = T
    cashflow = payoffs[:,-1].copy()

    # basic function factories
    def basis_funcs(S):
        """
        Return the polynomial basis [1, S, S^2] evaluated at the price vector S.
        S is expected to be a 1-D array with one entry per path.
        """
        S = np.asarray(S)
        return np.column_stack((np.ones_like(S), S, S**2))

    # backward recursion
    for t in range(nr_of_timesteps-1, -1, -1):

        Y = cashflow * discount_step
        S_t = paths[:, t, 1]  # use the simulated underlying price (column 1), not the step index
        X = basis_funcs(S_t)

        itm = (payoffs[:, t] > 0.0)     # array with 1's, where the option is In-The-Money (itm) and 0's elsewhere

        # If we don't have any itm-observations (or not enough for a reliable regression), we don't exercise anyone in this step and move the scheduled cashflows 
        # into the next period by discounting them
        if itm.sum() < X.shape[1]:
            cashflow = cashflow * discount_step
            continue

        # Filtering X & Y for ITM-paths
        X_itm = X[itm, :]       # shape (n_itm, n_basis)
        Y_itm = Y[itm]          # shape (n_itm, )

        # Safety check (ensuring we have more rows than columns)
        if X_itm.shape[0] <= X_itm.shape[1]:
            cashflow = cashflow * discount_step
            continue

        beta, *_ = np.linalg.lstsq(X_itm, Y_itm, rcond = None)

        # Estimate the continuation value of the option, using the calculated regression coefficient beta
        continuation_value = X.dot(beta)
        continuation_value[~itm] = np.inf       # Prevent early exercise of OTM paths by setting the continuation value against infinity

        # Exercise decision: exercise if immediate payoff is strictly greater than estimated continuation value
        immediate = payoffs[:,t]   
        exercise_now = immediate > continuation_value           # returns array with boolean values

        cashflow = np.where(exercise_now, immediate, cashflow)  # Checks each path in exercise_now (boolean_array) and if True (meaning the option gets exercised), the cashflow gets set to the immediate exercise value

        cashflow = cashflow * discount_step

    price = np.mean(cashflow)                   # Since we discounted our cashflow in every timestep, we don't need to discount the average cashflow a second time

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

    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    
    vega = S_0 * np.exp(-q * T) * nd1 * np.sqrt(T)
    
    term1 = -(S_0 * np.exp(-q * T) * nd1 * sigma) / (2 * np.sqrt(T))
    
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    
    if option_type == "call":
        delta = disc_q * Nd1
        rho = K * T * disc_r * Nd2
        term2 = q * S_0 * np.exp(-q * T) * Nd1
        term3 = r * K * np.exp(-r * T) * Nd2
        theta = term1 - term2 - term3
    else:
        delta = disc_q * (Nd1 - 1.0)
        rho = -K * T * disc_r * norm.cdf(-d2)
        term2 = q * S_0 * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = term1 + term2 + term3
    
    gamma = disc_q * nd1 / (S_0 * sigma * np.sqrt(T))
    
    return float(delta), float(gamma), float(rho), float(theta), float(vega)


# 5) Final Option Calculator

def option_calculator(file):
    """
    Final Option Calculator

    - Reads params via transform_input(file)
    - Routes pricing method based on exercise_type
    - Returns a dict with analytic and numeric prices + Greeks and (optionally) params
    - If output filename provided in params, writes CSV/JSON; otherwise prints (if verbose=True)
    """
    params = transform_input(file)
    exercise_type = str(params.get("exercise_type", "")).lower()
    option_type = str(params.get("option_type", "")).lower()
    output_file = params.get("filename")

    # placeholders (None = not computed)
    option_price_analytical = None
    option_price_mc = None

    # dispatch pricing
    if exercise_type == "european":
        # Analytic Black-Scholes pricing
        option_price_analytical = black_scholes(params)
        # Monte-Carlo pricing for direct comparison
        option_price_mc = mc_pricing_basic(params)

    elif exercise_type == "binary":
        option_price_analytical = black_scholes(params)
        option_price_mc = mc_pricing_basic(params)

    elif exercise_type in ("asian", "barrier"):
        option_price_mc = mc_pricing_basic(params)

    elif exercise_type == "american":
        # Use LSM for American pricing (Monte-Carlo)
        option_price_mc = longstaff_schwartz_pricing_american(params)
        # Also compute European/BS price as reference (if possible, i.e. if q = 0)
        if params["q"] == 0: 
            option_price_analytical = black_scholes(params)
        else:
            option_price_analytical = None

    else:
        raise ValueError(f"Unsupported exercise_type '{exercise_type}'. Supported: european, american, asian, barrier, binary")

    # Greeks (BS-based greeks; may be NaN for some inputs)
    try:
        delta, gamma, rho, theta, vega = bs_greeks(params)
    except Exception:
        delta = gamma = rho = theta = vega = float("nan")

    # Build output dict
    output = {
        "Exercise type": exercise_type,
        "Option type": option_type,
        "Option Price (Analytical (BS/EU))": option_price_analytical,
        "Option Price (Numerical (MC))": option_price_mc,
        "Delta": delta,
        "Gamma": gamma,
        "Rho": rho,
        "Theta": theta,
        "Vega": vega,
    }

    # Write to file or print
    if output_file:
        output_path = Path(output_file)
        suffix = output_path.suffix.lower()
        if suffix == ".csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                for key, value in output.items():
                    writer.writerow([key, value])
        elif suffix == ".json":
            # convert non-serializable values safely
            with open(output_path, "w") as f:
                json.dump(output, f, indent=4, default=str)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
    else:
        print("Exercise type:", exercise_type)
        print("Option type:", option_type)
        print("Option Price (Analytical (BS/EU)):", option_price_analytical)
        print("Option Price (Numerical (MC)):", option_price_mc)
        print("Delta:", delta)
        print("Gamma:", gamma)
        print("Rho:", rho)
        print("Theta:", theta)
        print("Vega:", vega)

    return output

filename_csv = os.path.join(os.getcwd(),'try.csv')
print(option_calculator(filename_csv))
