import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_cohen_kappa(ground_truth_df, llm_results_df, dimensions):
   
    cohen_kappa_results = {}

    for dim in dimensions:
        if dim not in ground_truth_df.columns or dim not in llm_results_df.columns:
            raise ValueError(f"Column {dim} is missing in either ground truth or LLM results")

       
        llm_ratings = llm_results_df[dim]
        ground_truth_ratings = ground_truth_df[dim]

        # Calculate Cohen's Kappa 
        cohen_kappa_results[dim] = cohen_kappa_score(llm_ratings, ground_truth_ratings)

    return cohen_kappa_results


def save_cohen_kappa_results_to_csv(cohen_kappa_results, filepath, model_name, setting, decimal_places=2):
  
   
    results_df = pd.DataFrame([cohen_kappa_results])

    
    results_df.insert(0, "Model", model_name)
    results_df.insert(1, "Setting", setting)

    # Round values for better readability
    results_df = results_df.round(decimal_places)

   
    results_df.to_csv(filepath, index=False)
    print(f"Cohen's Kappa results saved to {filepath}")
