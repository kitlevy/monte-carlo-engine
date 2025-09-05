from engine import MonteCarloEngine

engine = MonteCarloEngine(S0=100, r=0.05, sigma=0.2, T=1, steps=252, paths=100000)
call_price = engine.price_european_call(K=105)
put_price = engine.price_european_put(K=95)
print(f"Call: {call_price:.2f}, Put: {put_price:.2f}")
