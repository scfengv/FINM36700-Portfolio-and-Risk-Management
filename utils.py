import numpy as np
import pandas as pd

def CalcVar(returns: pd.Series):
    return returns.quantile(0.05)

def CalcCVar(returns: pd.Series):
    return returns[returns <= CalcVar(returns)].mean()

def CalcLevelReturns(returns: pd.Series, annualizedFactor: int) -> float:
    """_summary_
    Args:
        returns (pd.Series): Times Series returns data (monthly / weekly / daily)
            - Date as index
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252
        
    Returns:
        Annualized mean of Level Return
    """
    return returns.mean() * annualizedFactor

def CalcLogReturns(returns: pd.Series, annualizedFactor: int) -> float:
    """_summary_

    Args:
        returns (pd.Series): Times Series returns data (monthly / weekly / daily)
        annualizedFactor (int): monthly = 12; weekly = 52; daily = 252

    Returns:
        Annualized mean of Log Return
    """
    logReturns = np.log1p(1 + returns)
    return logReturns.mean() * annualizedFactor