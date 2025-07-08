import pandas as pd

def compute_accuracy(ground_truth_df, llm_results_df, dimensions):
    """
    Parameters:
        ground_truth_df 
        llm_results_df 
        dimensions (list)

    """
   
    if 'argQ_id' in ground_truth_df.columns and 'argQ_id' in llm_results_df.columns:
        ground_truth_df = ground_truth_df.set_index('argQ_id')
        llm_results_df = llm_results_df.set_index('argQ_id')

  
    llm_results_df = llm_results_df.reindex(ground_truth_df.index)
    
    accuracy_results = {}
    
    for dim in dimensions:
        if dim not in ground_truth_df.columns or dim not in llm_results_df.columns:
            raise ValueError(f"Column {dim} is missing in either ground truth or LLM results")
        
        correct_predictions = (ground_truth_df[dim] == llm_results_df[dim]).sum()
        total_predictions = len(ground_truth_df)
        
        accuracy_results[dim] = correct_predictions / total_predictions
    

    accuracy_results["Overall_Accuracy"] = sum(accuracy_results.values()) / len(dimensions)
    
    return accuracy_results


def save_accuracy_results_to_csv(accuracy_results, filepath, model_name, setting, decimal_places=2):

    results_df = pd.DataFrame([accuracy_results])
    

    results_df.insert(0, "Model", model_name)
    results_df.insert(1, "Setting", setting)
    
 
    results_df = results_df.round(decimal_places)
    

    results_df.to_csv(filepath, index=False)
    print(f"Accuracy results saved to {filepath}")
