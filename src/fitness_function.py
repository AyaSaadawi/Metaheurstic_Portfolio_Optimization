import numpy as np

def fitness_function(weights, returns_df):
    weights = np.array(weights)
    print("Weights:", weights)
    print("Type of elements in weights:", [type(w) for w in weights])
    weights = np.clip(weights, 0.0, 1.0)
    weights /= np.sum(weights)

    # Extract return matrix
    return_matrix = returns_df.values  # shape: (n_days, n_assets)

    portfolio_returns = np.dot(return_matrix, weights)

    # Sharpe ratio
    mean_ret = np.mean(portfolio_returns)
    std_ret = np.std(portfolio_returns)
    
    if std_ret == 0:
        return float('inf')
    sharpe_ratio = mean_ret / std_ret if std_ret != 0 else 0
    return -sharpe_ratio 