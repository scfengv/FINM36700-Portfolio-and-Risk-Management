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
    plt.show()
    
def Calc_TangencyWeights(excessReturns: pd.DataFrame, annualizedFactor: int):
    """
    Tangency Weights = scaling * (Inverse of Covariance Matrix @ Annualized Excess Return)
    
    Also for Mean-Variance

    Args:
        excessReturns (pd.DataFrame)
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
        returns (pd.DataFrame)
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        Risk Parity Weights
    """
    vol = returns.var() * annualizedFactor
    invVol = 1 / vol
    return np.array(invVol / invVol.sum())

def Calc_MeanStdSharpe(returns: pd.DataFrame, weights: pd.Series, annualizedFactor: int) -> tuple[float, float, float]:
    """
    Args:
        returns (pd.DataFrame)
        weights (pd.Series)
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        tuple[float, float, float]: (mean, std, sharpe)
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
        returns (pd.DataFrame)
        weights (pd.Series)
        targetReturn (float): per month
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Leverage Ratio (float)
    """
    monthlyReturn = returns @ weights
    currentMean = monthlyReturn.mean() # Monthly Return w/o leverage
    
    leverageRatio = targetReturn / currentMean
    return leverageRatio

def Calc_MaxDrawdown(returns) -> float:
    """
    Args:
        returns (pd.DataFrame, pd.Series)

    Returns:
        Max Drawdown
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    maxDrawdown = drawdown.min()
    return maxDrawdown

def Calc_SkewKurt(returns) -> tuple[float, float]:
    """
    Args:
        returns (pd.DataFrame, pd.Series)

    Returns:
        tuple[float, float]: (skew, kurt)
    """
    return (returns.skew(), returns.kurt())

def Calc_Beta_TreynorRatio_InfoRatio(returns, benchmark: pd.Series, annualizedFactor: int) -> tuple[list[float], list[float], list[float]]:
    """
    beta: y_hat = alpha + (beta1 * benchmark) + residual
     
    Treynor Ratio = Mean of Excess Return / beta
    
    Information Ratio = alpha / std(residual)
    
    Args:
        returns (pd.DataFrame, pd.Series)
        benchmark (pd.Series): Market data (e.g. S&P 500)
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Beta (list[float]), Treynor Ratio (list[float]), Information Ratio (list[float])
    """
    
    marketBeta, treynorRatio, infoRatio = list(), list(), list()
    x = sm.add_constant(benchmark)
    
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    
    for col in returns.columns:
        y = returns[col].ffill()
        model = sm.OLS(y, x).fit()
        
        alpha, beta = model.params
        epsilon = model.resid.std() * np.sqrt(annualizedFactor)
        
        treynor = y.mean() * annualizedFactor / beta
        info = alpha / epsilon

        marketBeta.append(beta)
        treynorRatio.append(treynor)
        infoRatio.append(info)
        
    return (marketBeta, treynorRatio, infoRatio)
        