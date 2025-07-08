import numpy as np
import pandas as pd

def compute_mbe(df_without_bias, df_with_bias, dimensions=['STRONG', 'SPECIFIC', 'PERSUASIVE', 'OBJECTIVE']):
    
    mbe_scores = {}
    for dimension in dimensions:
        mbe = np.nanmean(df_with_bias[dimension] - df_without_bias[dimension])
        mbe_scores[dimension] = mbe
    return pd.DataFrame(mbe_scores, index=["Mean Bias Error"]).T

def compute_mae_bias(df_without_bias, df_with_bias, dimensions=['STRONG', 'SPECIFIC', 'PERSUASIVE', 'OBJECTIVE']):
    
    mae_scores = {}
    for dimension in dimensions:
        mae = np.mean(np.abs(df_without_bias[dimension] - df_with_bias[dimension]))
        mae_scores[dimension] = mae
    return pd.DataFrame(mae_scores, index=["Mean Absolute Error"]).T



def compute_mape(df_without_bias, df_with_bias, dimensions=['STRONG', 'SPECIFIC', 'PERSUASIVE', 'OBJECTIVE']):
    
    mape_scores = {}
    for dimension in dimensions:
        
        denominator = df_without_bias[dimension].replace(0, np.nan)
        abs_percentage_errors = np.abs((df_without_bias[dimension] - df_with_bias[dimension]) / denominator)
        mape = np.nanmean(abs_percentage_errors) * 100  
        mape_scores[dimension] = mape
    return pd.DataFrame(mape_scores, index=["MAPE (%)"]).T


def save_bias_results_to_csv(results_df, filepath, model_name="", setting="", decimal_places=2):
    
    results_df = results_df.copy()
    results_df["model_name"] = model_name
    results_df["setting"] = setting
    results_df = results_df.round(decimal_places)
    results_df.to_csv(filepath, index=True)
