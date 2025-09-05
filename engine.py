import numpy as np

class MonteCarloEngine:
    def __init__(self, S0, r, sigma, T, steps, paths, seed=None):
        """
        S0: initial asset price
        r: risk-free rate
        sigma: volatility
        T: time to maturity (years)
        steps: number of time steps per path
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
        # Precompute drift and diffusion
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        # Simulate random increments
        Z = np.random.randn(self.paths, self.steps)
        increments = drift + diffusion * Z
        # Convert increments to price paths
        log_paths = np.cumsum(increments, axis=1)
        S_paths = self.S0 * np.exp(log_paths)
        # Include initial price
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

