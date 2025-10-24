import numpy as np
import pandas as pd
import math
from scipy.stats import norm


def black_scholes(S_0, K, T, r, sigma, option_type = 'call'):
    '''
    Analytical Solution for the Price of European Options.
    Takes as inputs, the current price of the underlying, the strike price of the option, expiration, 
    interest rate r, implied volatility (sigma), and the option type (call/put), and returns the "fair"
    option price. 
    '''

    option_type = option_type.lower()

    # Define Helper Terms
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    if option_type == 'call':
        p = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    elif option_type == 'put':
        p = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)

    else: 
        raise ValueError('Option type must be "Call" or "Put"')

    return p


def timestep(start_price, r, q, sigma, delta_t):
    '''
    Simulates one timestep in the Monte-Carlo simulation. Takes a given start_price at the beginning of the timestep 
    and transforms it into a random price (acc. to GBM) after the timestep.
    '''
    return start_price * np.exp((r-q-0.5*sigma**2)*delta_t + sigma * np.sqrt(delta_t)*np.random.normal(0,1))


def simulation_run(timesteps, start_price, expiration, r, q, sigma):
    '''
    One full pricing simulation for n timesteps. Takes a given start_price as well as the number of timesteps, and simulates
    an asset's random price development (acc. to GBM) over the n timesteps. Returns an array of the format: [[timestep, price]]
    '''
    prices = [[0, start_price]]
    delta_t = expiration/timesteps
    S_i = start_price
    for i in range(timesteps):
        S_i = timestep(S_i, r, q, sigma, delta_t)
        prices.append([i+1, S_i])
    return prices


def MonteCarlo(simulations, timesteps, start_price, expiration, r, q, sigma):
    '''
    Monte-Carlo simulation for an Asset's price development, given a start_price, the assets volatility, current interest rate, dividend yield, expiration
    the number of timesteps per simulation and the total number of simulations. Simulates M different price developments with n timesteps each. 
    Each price development follows a Stochastic Wiener process (i.e. Geometric Brownian Motion).
    '''
    list_of_prices = []
    for _ in range(simulations):
        list_of_prices.append(simulation_run(timesteps, start_price, expiration, r, q, sigma))
    return list_of_prices


# Testrun
# print(MonteCarlo(10, 10, 100, 1/12, 0.04, 0.01, 0.30))