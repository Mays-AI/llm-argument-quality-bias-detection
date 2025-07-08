

import pandas as pd

def create_composite_key(df, key_cols=None):
    """
    Create a composite key for each row by concatenating specified columns.
    """
    if key_cols is None:
        key_cols = ['premise_texts', 'argQ_id', 'claim_text']
    return df[key_cols[0]].astype(str) + "_" + df[key_cols[1]].astype(str) + "_" + df[key_cols[2]].astype(str)

def rename_annotation_columns(df, dimensions, suffix):
    
    rename_dict = {dim: f"{dim}{suffix}" for dim in dimensions}
    return df.rename(columns=rename_dict)

def compute_bias(merged_df, dimensions):
    
    def bias_commentary(delta):
        if delta > 0:
            return " (positively biased)"
        elif delta < 0:
            return " (negatively biased)"
        else:
            return " (neutral)"
    
    for dim in dimensions:
        merged_df[f'{dim}_female_llm_delta'] = merged_df[f'{dim}_llm_female'] - merged_df[f'{dim}_llm']
        merged_df[f'{dim}_male_llm_delta'] = merged_df[f'{dim}_llm_male'] - merged_df[f'{dim}_llm']
        
        merged_df[f'{dim}_llm_female_evals'] = merged_df[f'{dim}_llm_female'].astype(str) + \
            merged_df[f'{dim}_female_llm_delta'].apply(bias_commentary)
        merged_df[f'{dim}_llm_male_evals'] = merged_df[f'{dim}_llm_male'].astype(str) + \
            merged_df[f'{dim}_male_llm_delta'].apply(bias_commentary)
    return merged_df

def merge_annotations(original_path, llm_path, female_path, male_path, dimensions=None, usecols=None):
    """
    Merge annotation CSV files and compute bias commentary.
    
    Parameters:
        original_path (str): Path to the original annotations CSV.
        llm_path (str): Path to the LLM annotations CSV.
        female_path (str): Path to the female annotations CSV.
        male_path (str): Path to the male annotations CSV.
        dimensions (list): List of annotation dimensions. Defaults to 
                           ["SPECIFIC", "PERSUASIVE", "STRONG", "OBJECTIVE"].
        usecols (list): List of columns to load. Defaults to
                        ['argQ_id', 'claim_text', 'premise_texts'] + dimensions.
                        
    Returns:
        pd.DataFrame: The final merged DataFrame with bias commentary columns.
    """
    if dimensions is None:
        dimensions = ["SPECIFIC", "PERSUASIVE", "STRONG", "OBJECTIVE"]
    if usecols is None:
        usecols = ['argQ_id', 'claim_text', 'premise_texts'] + dimensions

    # Load CSV files
    df_orig = pd.read_csv(original_path, usecols=usecols)
    df_llm = pd.read_csv(llm_path, usecols=usecols)
    df_female = pd.read_csv(female_path, usecols=usecols)
    df_male = pd.read_csv(male_path, usecols=usecols)
    
    # Create composite keys and drop duplicates
    for df in [df_orig, df_llm, df_female, df_male]:
        df['join_key'] = create_composite_key(df)
        df.drop_duplicates(subset='join_key', inplace=True)
    
    # Rename annotation columns for clarity
    df_orig = rename_annotation_columns(df_orig, dimensions, '_orig')
    df_llm = rename_annotation_columns(df_llm, dimensions, '_llm')
    df_female = rename_annotation_columns(df_female, dimensions, '_llm_female')
    df_male = rename_annotation_columns(df_male, dimensions, '_llm_male')
    
    # Merge DataFrames on join_key
    merged = df_orig.merge(df_llm, on='join_key', how='inner', suffixes=('', '_llm'))
    for col in ['argQ_id', 'premise_texts', 'claim_text']:
        dup = col + '_llm'
        if dup in merged.columns:
            merged.drop(columns=dup, inplace=True)
    
    merged = merged.merge(df_female, on='join_key', how='inner', suffixes=('', '_llm_female'))
    for col in ['argQ_id', 'premise_texts', 'claim_text']:
        dup = col + '_llm_female'
        if dup in merged.columns:
            merged.drop(columns=dup, inplace=True)
    
    merged = merged.merge(df_male, on='join_key', how='inner', suffixes=('', '_llm_male'))
    for col in ['argQ_id', 'premise_texts', 'claim_text']:
        dup = col + '_llm_male'
        if dup in merged.columns:
            merged.drop(columns=dup, inplace=True)
            
 
    merged.drop(columns='join_key', inplace=True)
    

    merged = compute_bias(merged, dimensions)
    
    return merged
