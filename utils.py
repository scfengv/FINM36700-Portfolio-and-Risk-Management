import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

