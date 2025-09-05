import numpy as np

#things to improve:
#separate path simulation from payoff evaluation
#allow user to define payoff function
#make more reusable for different derivatives
#option for antithetic variance? read up on it
#maybe try out numba?

class MonteCarloEngine:
    def __init__(self, S0, r, sigma, T, steps=252, paths, seed=None):
        """
        S0: initial asset price
        r: risk-free rate
        sigma: volatility
        T: time to maturity (years)
        steps: number of time steps per path (default 252 trading days a year)
        paths: number of Monte Carlo paths
        seed: optional random seed
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.paths = paths
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_paths(self):
        dt = self.T / self.steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        #simulate random increments
        Z = np.random.randn(self.paths, self.steps)
        increments = drift + diffusion * Z
        #convert increments to price paths
        log_paths = np.cumsum(increments, axis=1)
        S_paths = self.S0 * np.exp(log_paths)
        S_paths = np.hstack((self.S0 * np.ones((self.paths, 1)), S_paths))
        return S_paths
    
    def price_european_call(self, K):
        paths = self.simulate_paths()
        payoff = np.maximum(paths[:, -1] - K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price
    
    def price_european_put(self, K):
        paths = self.simulate_paths()
        payoff = np.maximum(K - paths[:, -1], 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price

    #asian options
    def price_asian_call(self, K):
        paths = self.simulate_paths()
        avg_price = np.mean(paths[:, 1:], axis=1)  #exclude S0
        payoff = np.maximum(avg_price - K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price
    
    #greeks using finite diff
    def delta_fd(self, K, h=0.01):
        price_up = MonteCarloEngine(self.S0 + h, self.r, self.sigma, self.T, self.steps, self.paths).price_european_call(K)
        price_down = MonteCarloEngine(self.S0 - h, self.r, self.sigma, self.T, self.steps, self.paths).price_european_call(K)
        return (price_up - price_down) / (2*h)

if __name__ == '__main__':
    engine = MonteCarloEngine(S0=100, r=0.05, sigma=0.2, T=1, steps=252, paths=100000)
    call_price = engine.price_european_call(K=105)
    #call_price = engine.price_asian_call(K=105)
    put_price = engine.price_european_put(K=95)
    print(f"Call: {call_price:.2f}, Put: {put_price:.2f}")
