import numpy as np

#improvements made:
#separate path simulation from payoff evaluation
#allow user to define payoff function
#make more reusable for different derivatives
#option for antithetic variates? read up on it
#reuse same numbers for up and down
#not done:
#maybe try out numba?

class MonteCarloEngine:
    def __init__(self, S0, r, sigma, T, steps=252, paths=100000, seed=None, antithetic=False):
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
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed) #local rng for reproducible tests

    def simulate_paths(self):
        dt = self.T / self.steps
        drift = (self.r - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        Z = self.rng.standard_normal((self.paths, self.steps))
        if self.antithetic:
            Z = np.vstack((Z, -Z))

        increments = drift + diffusion * Z
        log_paths = np.cumsum(increments, axis=1)
        S_paths = self.S0 * np.exp(log_paths)
        n_paths = S_paths.shape[0]
        S_paths = np.hstack((self.S0 * np.ones((n_paths, 1)), S_paths))
        return S_paths

    def price_option(self, payoff_func):
        """
        generic option pricing using a user-defined payoff function.
        payoff_func: function accepting (paths_array) -> payoff_array
        """
        paths = self.simulate_paths()
        payoff = payoff_func(paths)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    #default payoff funcs
    @staticmethod
    def european_call(K):
        return lambda paths: np.maximum(paths[:, -1] - K, 0)
    
    @staticmethod
    def european_put(K):
        return lambda paths: np.maximum(K - paths[:, -1], 0)
    
    @staticmethod
    def asian_call(K):
        return lambda paths: np.maximum(np.mean(paths[:, 1:], axis=1) - K, 0)
    
    @staticmethod
    def asian_put(K):
        return lambda paths: np.maximum(K - np.mean(paths[:, 1:], axis=1), 0)
    
    #greeks using finite diff
    def delta(self, K, h=0.01, payoff_func=None):
        if payoff_func is None:
            if K is None:
                raise ValueError("Either K or payoff_func must be provided")
            payoff_func = self.european_call(K)

        #reusing numbers for up and down to reduce MC noise
        base_paths = self.rng.standard_normal((self.paths, self.steps))
        if self.antithetic:
            base_paths = np.vstack((base_paths, -base_paths))

        def price_for_S0(S0_shift):
            dt = self.T / self.steps
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt)
            increments = drift + diffusion * base_paths
            log_paths = np.cumsum(increments, axis=1)
            S_paths = S0_shift * np.exp(log_paths)
            n_paths = S_paths.shape[0]
            S_paths = np.hstack((S0_shift * np.ones((n_paths, 1)), S_paths))
            payoff = payoff_func(S_paths)
            return np.exp(-self.r * self.T) * np.mean(payoff)

        price_up = price_for_S0(self.S0 + h)
        price_down = price_for_S0(self.S0 - h)
        return (price_up - price_down) / (2 * h)

    def gamma(self, K=None, h=0.01, payoff_func=None):
        if payoff_func is None:
            if K is None:
                raise ValueError("Either K or payoff_func must be provided")
            payoff_func = self.european_call(K)

        base_paths = self.rng.standard_normal((self.paths, self.steps))
        if self.antithetic:
            base_paths = np.vstack((base_paths, -base_paths))

        def price_for_S0(S0_shift):
            dt = self.T / self.steps
            drift = (self.r - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt)
            increments = drift + diffusion * base_paths
            log_paths = np.cumsum(increments, axis=1)
            S_paths = S0_shift * np.exp(log_paths)
            n_paths = S_paths.shape[0]
            S_paths = np.hstack((S0_shift * np.ones((n_paths, 1)), S_paths))
            payoff = payoff_func(S_paths)
            return np.exp(-self.r * self.T) * np.mean(payoff)

        price_up = price_for_S0(self.S0 + h)
        price = price_for_S0(self.S0)
        price_down = price_for_S0(self.S0 - h)
        return (price_up - 2 * price + price_down) / (h ** 2)

if __name__ == '__main__':
    engine = MonteCarloEngine(S0=100, r=0.05, sigma=0.2, T=1,
                          steps=252, paths=50000, antithetic=True)

    #european call
    call_price = engine.price_option(MonteCarloEngine.european_call(K=105))
    print(f"European Call Price: {call_price:.4f}")

    #asian call
    asian_price = engine.price_option(MonteCarloEngine.asian_call(K=105))
    print(f"Asian Call Price: {asian_price:.4f}")

    #compute Greeks
    delta_call = engine.delta(K=105)
    gamma_call = engine.gamma(K=105)
    print(f"Delta: {delta_call:.4f}, Gamma: {gamma_call:.4f}")
