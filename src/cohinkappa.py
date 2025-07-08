"""

This module provides functions to:
 Sample 20% of the data from a set of specified companies (5% each).

"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def create_sample(df, specified_companies, fraction=0.20, random_state=42):
    
    total_sample_size = int(len(df) * fraction)
    sample_size_per_company = int(total_sample_size / len(specified_companies))

    sample_df = pd.DataFrame()
    for company in specified_companies:
        company_data = df[df['company_name'] == company]
        # Ensure reproducible sampling
        sample = company_data.sample(n=sample_size_per_company, random_state=random_state)
        sample_df = pd.concat([sample_df, sample], ignore_index=True)

    return sample_df


def compute_random_cohens_kappa(sample_df, dimensions, random_seed=42):
    
    random.seed(random_seed)
    kappa_results = {}

    for dim in dimensions:
       
        r1, r2 = random.sample([0, 1, 2], 2)

        chosen_run_1 = []
        chosen_run_2 = []

        for _, row in sample_df.iterrows():
            run_list = row[dim]
            
            if isinstance(run_list, list) and len(run_list) == 3:
                chosen_run_1.append(run_list[r1])
                chosen_run_2.append(run_list[r2])
            else:
                
                chosen_run_1.append(np.nan)
                chosen_run_2.append(np.nan)

        
        run1_series = pd.Series(chosen_run_1)
        run2_series = pd.Series(chosen_run_2)

      
        valid_data = pd.concat([run1_series, run2_series], axis=1).dropna()
        runA_values = valid_data.iloc[:, 0].astype(int)
        runB_values = valid_data.iloc[:, 1].astype(int)

        # Calculate Cohen's Kappa
        kappa_value = cohen_kappa_score(runA_values, runB_values)

        # Store results
        kappa_results[dim] = {
            'Chosen Runs': f'{r1} vs {r2}',
            'Kappa': kappa_value
        }

    return kappa_results

def save_kappa_results_to_csv(kappa_results, filepath, model_name="", setting="", decimal_places=2):
    
    rows = []
    for dim, data in kappa_results.items():
        chosen_runs = data["Chosen Runs"]
        kappa_value = data["Kappa"]

        rows.append({
            "Model Name": model_name,
            "Setting": setting,
            "Dimension": dim,
            "Chosen Runs": chosen_runs,
            "Kappa": round(kappa_value, decimal_places)
        })

   
    df = pd.DataFrame(rows, columns=["Model Name", "Setting", "Dimension", "Chosen Runs", "Kappa"])


    df.to_csv(filepath, index=False)
    print(f"Saved Cohen's Kappa results to {filepath}")
