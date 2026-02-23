import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Step 1: Define tickers
tickers = ["AAPL", "MSFT", "SPY"]
weights = [0.30, 0.30, 0.40]

# Step 2: Download historical data
prices = yf.download(tickers, start="2023-01-01", end="2024-01-01", auto_adjust=True)["Close"]
print(prices.head())
# Step 2: Calculate daily returns
returns = prices.pct_change().dropna()
# Step 3: Portfolio daily returns (weighted)
portfolio_returns = (returns * weights).sum(axis=1)
# Step 4: Portfolio Metrics

avg_daily_return = portfolio_returns.mean()
daily_volatility = portfolio_returns.std()

annual_return = avg_daily_return * 252
annual_volatility = daily_volatility * (252 ** 0.5)
risk_free_rate = 0.04  # assume 4%
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

print("\nPortfolio Metrics:")
print("Average Daily Return:", avg_daily_return)
print("Daily Volatility:", daily_volatility)
print("Annual Return:", annual_return)
print("Annual Volatility:", annual_volatility)



print("\nPortfolio Daily Returns:")
print(portfolio_returns.head())

print("\nDaily Returns:")
print(returns.head())
import numpy as np

print("\nRunning Portfolio Optimization...")

num_portfolios = 5000
results = []
weights_record = []

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

for _ in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

    results.append(sharpe)
    weights_record.append(weights)

max_sharpe_index = np.argmax(results)
optimal_weights = weights_record[max_sharpe_index]

print("\nOptimal Portfolio Found:")
print("Weights:", dict(zip(tickers, optimal_weights)))
print("Expected Return:", np.dot(optimal_weights, mean_returns))
print("Expected Volatility:", np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))))
print("Sharpe Ratio:", results[max_sharpe_index])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Efficient Frontier (Monte Carlo) ---
# Using historical daily returns to estimate mean + covariance
mean_daily = returns.mean()
cov_daily = returns.cov()

N_PORTFOLIOS = 5000
results = []
weight_list = []

for _ in range(N_PORTFOLIOS):
    w = np.random.random(len(tickers))
    w = w / w.sum()

    port_return = np.dot(w, mean_daily) * 252
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_daily * 252, w)))
    sharpe = (port_return - risk_free_rate) / port_vol

    results.append([port_return, port_vol, sharpe])
    weight_list.append(w)

results = np.array(results)

# Find best Sharpe portfolio
max_sharpe_idx = np.argmax(results[:, 2])
best_return, best_vol, best_sharpe = results[max_sharpe_idx]
best_weights = weight_list[max_sharpe_idx]

print("\n=== Efficient Frontier Best Portfolio ===")
print("Best Sharpe:", best_sharpe)
print("Return:", best_return)
print("Volatility:", best_vol)
print("Weights:")
for t, w in zip(tickers, best_weights):
    print(f"  {t}: {w:.2%}")

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], s=10)
plt.scatter(best_vol, best_return, s=200, marker="*", edgecolors="k")
plt.title("Efficient Frontier (Monte Carlo) — Sharpe Maximization")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Return")
plt.colorbar(label="Sharpe Ratio")
plt.show()
# Step 5: Plot cumulative portfolio returns

cumulative_returns = (1 + portfolio_returns).cumprod()

plt.figure(figsize=(10,6))
plt.plot(cumulative_returns, label="Optimized Portfolio")
plt.title("Portfolio Growth Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()