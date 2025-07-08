import pandas as pd
import ast
from statsmodels.stats.inter_rater import fleiss_kappa

def parse_run_list(x):
  
    if isinstance(x, str):
        x = x.strip()
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x

def get_score_frequencies(score_list):
  
    frequencies = [0, 0, 0]  
    
    if isinstance(score_list, list):
        for score in score_list:
            if score == 0:
                frequencies[0] += 1
            elif score == 1:
                frequencies[1] += 1
            elif score == 2:
                frequencies[2] += 1
    return frequencies

def compute_fleiss_kappa(df, dimensions):
 
    for dim in dimensions:
        df[dim] = df[dim].apply(parse_run_list)

    df_frequencies = pd.DataFrame({
        f"{dim}_freq": df[dim].apply(get_score_frequencies) for dim in dimensions
    })

    for dim in dimensions:
        df_frequencies[[f"{dim}_0", f"{dim}_1", f"{dim}_2"]] = pd.DataFrame(df_frequencies[f"{dim}_freq"].tolist(), index=df_frequencies.index)

  
    df_frequencies = df_frequencies.drop(columns=[f"{dim}_freq" for dim in dimensions])


    df_frequencies["sum_frequencies"] = df_frequencies[[f"{dim}_0", f"{dim}_1", f"{dim}_2"]].sum(axis=1)

    df_frequencies_cleaned = df_frequencies[df_frequencies["sum_frequencies"] == 3].drop(columns=["sum_frequencies"])

    print(f"Removed {len(df_frequencies) - len(df_frequencies_cleaned)} rows with inconsistent annotation counts.")

    kappa_results = {
        dim: fleiss_kappa(df_frequencies_cleaned[[f"{dim}_0", f"{dim}_1", f"{dim}_2"]].to_numpy(), method='fleiss')
        for dim in dimensions
    }

    return kappa_results

def save_fleiss_kappa_results_to_csv(kappa_results, filepath, model_name, setting, decimal_places=3):
  
 
    results_df = pd.DataFrame([kappa_results])

    results_df.insert(0, "Model", model_name)
    results_df.insert(1, "Setting", setting)


    results_df = results_df.round(decimal_places)

    results_df.to_csv(filepath, index=False)
    print(f"Fleiss' Kappa results saved to {filepath}")
