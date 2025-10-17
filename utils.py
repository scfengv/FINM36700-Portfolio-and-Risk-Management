import pandas as pd

def CalcVar(returns: pd.Series):
    return returns.quantile(0.05)

def CalcCVar(returns: pd.Series):
    return returns[returns <= CalcVar(returns)].mean()

