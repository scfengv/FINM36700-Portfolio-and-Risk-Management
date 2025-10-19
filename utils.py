import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.optimize import minimize

def Calc_Var(returns: pd.Series):
    return returns.quantile(0.05)

def Calc_CVar(returns: pd.Series):
    return returns[returns <= Calc_Var(returns)].mean()

def Calc_LevelReturns(returns: pd.Series, annualizedFactor: int) -> float:
    """
    Args:
        returns (pd.Series): Times Series returns data (monthly / weekly / daily)
            - Date as index
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Annualized mean of Level Return
    """
    return returns.mean() * annualizedFactor

def Calc_LogReturns(returns: pd.Series, annualizedFactor: int) -> float:
    """
    Args:
        returns (pd.Series): Times Series returns data (monthly / weekly / daily)
            - Date as index
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        Annualized mean of Log Return
    """
    logReturns = np.log1p(1 + returns)
    return logReturns.mean() * annualizedFactor

def Plot_CorrHeatmap(returns: pd.DataFrame):
    corrMatrix = returns.corr()
    plt.figure(figsize = (12, 10))
    sns.heatmap(
        corrMatrix, annot = True, fmt = ".2f", cmap = "coolwarm", center = 0,
        square = True, linewidths = 1
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
def Calc_TangencyWeights(excessReturns: pd.DataFrame, annualizedFactor: int):
    """
    Tangency Weights = scaling * (Inverse of Covariance Matrix @ Annualized Excess Return)
    
    Also for Mean-Variance

    Args:
        excessReturns (pd.DataFrame):
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Tangency Weights
    """
    covMat = excessReturns.cov() * annualizedFactor # Covariance matrix * Annual factor
    covInv = np.linalg.inv(covMat) # Sigma^(-1)
    mu = excessReturns.mean() * annualizedFactor # Annualized Mean Excess Return
    scaling = 1 / (np.transpose(np.ones(len(excessReturns.columns))) @ covInv @ mu)
    tangencyWeights = scaling * (covInv @ mu)
    return tangencyWeights

def Calc_EqWeights(returns: pd.DataFrame):
    numAsset = len(returns.columns)
    return np.array([1 / numAsset for _ in range(numAsset)])

def Calc_RiskParityWeights(returns: pd.DataFrame, annualizedFactor: int):
    """
    w(RP) = 1 / var

    Args:
        returns (pd.DataFrame):
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        Risk Parity Weights
    """
    vol = returns.var() * annualizedFactor
    invVol = 1 / vol
    return np.array(invVol / invVol.sum())

def Calc_MeanStdSharpe_Stock(returns: pd.Series or pd.DataFrame, annualized_factor: int) -> tuple[float, float, float]:
    """
    Mean, STD, Sharpe of Single Stock
    
    Args:
        returns (pd.Seriesorpd.DataFrame): Stocks
        annualized_factor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        tuple: (mean, std, sharpe)
    """
    mean = returns.mean() * annualized_factor
    std = returns.std() * np.sqrt(annualized_factor)
    sharpe = mean / std
    return (mean, std, sharpe)

def Calc_MeanStdSharpe_Portfolio(returns: pd.DataFrame, weights: pd.Series, annualizedFactor: int) -> tuple[float, float, float]:
    """
    Args:
        returns (pd.DataFrame): Portfolio
        weights (pd.Series):
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        tuple: (mean, std, sharpe)
    """
    portfolioReturn = returns @ weights
    mean = portfolioReturn.mean() * annualizedFactor
    std = portfolioReturn.std() * np.sqrt(annualizedFactor)
    sharpe = mean / std
    return (mean, std, sharpe)

def Calc_LeverageRatio_Monthly(returns: pd.DataFrame, weights: pd.Series, targetReturn: float, annualizedFactor = 12) -> float:
    """
    Calculate Leverage Ratio of portfolio to meet "Monthly" Target Return

    Args:
        returns (pd.DataFrame):
        weights (pd.Series):
        targetReturn (float): per month
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Leverage Ratio (float)
    """
    monthlyReturn = returns @ weights
    currentMean = monthlyReturn.mean() # Monthly Return w/o leverage
    
    leverageRatio = targetReturn / currentMean
    return leverageRatio

def Calc_MaxDrawdown(returns) -> tuple[float, list[float]]:
    """
    Args:
        returns (pd.DataFrame, pd.Series):

    Returns:
        Tuple: (Max Drawdown, Drawdown (Time Series))
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    maxDrawdown = drawdown.min()
    return (maxDrawdown, drawdown)

def Calc_SkewKurt(returns) -> tuple[float, float]:
    """
    Skewness and Kurtosis
    
    Args:
        returns (pd.DataFrame, pd.Series):

    Returns:
        Tuple: (Skewness, Kurtosis)
    """
    return (returns.skew(), returns.kurt())

def Calc_Beta_TreynorRatio_InfoRatio_RSquared_TrackingError(y: pd.Series, x: pd.Series or pd.DataFrame, annualized_factor: int):
    """
    beta: y_hat = alpha + (beta1 * benchmark) + residual
     
    Treynor Ratio = Mean of Excess Return / beta
    
    Information Ratio = alpha / std(residual)
    
    Args:
        y (pd.Series): The dependent variable (e.g., asset return).
        x (pd.Series or pd.DataFrame): The independent variable(s).
                                     - pd.Series for simple regression (Case 1).
                                     - pd.DataFrame for multiple regression (Case 2).
        annualized_factor (int): e.g., monthly = 12
        
    Returns:
        Tuple: 
        - Case 1 (Simple): (Beta, Treynor Ratio, Information Ratio, R squared, Tracking Error)
        - Case 2 (Multiple): (Betas, None, Information Ratio, R squared, Tracking Error)
    """
    y, x = y.ffill(), x.ffill()
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()

    alpha_monthly = model.params["const"]
    alpha_annual = alpha_monthly * annualized_factor
    
    # Tracking Error
    epsilon_monthly = model.resid.std()
    epsilon_annual = epsilon_monthly * np.sqrt(annualized_factor)
    
    info_ratio = alpha_annual / epsilon_annual

    # Case 1: Simple Regression
    if isinstance(x, pd.Series):
        beta = model.params.drop("const").iloc[0]
        y_mean_annual = y.mean() * annualized_factor
        treynor_ratio = y_mean_annual / beta
        r_squared = model.rsquared
        
        return (beta, treynor_ratio, info_ratio, r_squared, epsilon_annual)
    
    # Case 2: Multiple Regression
    else:
        betas = model.params.drop("const") 
        r_squared = model.rsquared
        
        # Treynor Ratio is not well-defined for multiple betas.
        return (betas, None, info_ratio, r_squared, epsilon_annual)

def Calc_CumulativeReturn(returns: pd.Series) -> list[float]:
    """
    Cumulative Return for [0.1, -0.05, 0.02]
    
    `(1 + returns).cumprod()`: [1.1, 1.045, 1.0659]
    
    `(1 + returns).prod()`: 1.0659

    Args:
        returns (pd.Series):

    Returns:
        list[float]: a time series of cumulative returns
    """
    return 100 * (1 + returns).cumprod()

