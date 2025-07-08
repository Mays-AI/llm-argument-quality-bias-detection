import pandas as pd
import os
import ast

def is_valid_integer(value):
    
    try:
        int(value)  
        return True
    except ValueError:
        return False

def clean_and_validate_data(input_path, output_path):
 
    
    print(f"\nProcessing file: {input_path}")

    df = pd.read_csv(input_path)

    target_columns = ['STRONG', 'SPECIFIC', 'PERSUASIVE', 'OBJECTIVE']

    df[target_columns] = df[target_columns].apply(pd.to_numeric, errors='coerce')

   
    df[target_columns] = df[target_columns].round().astype('Int64')

    df.to_csv(output_path, index=False)
    print(f" Cleaned data saved to: {output_path}")

    valid_integers = df[target_columns].applymap(lambda x: isinstance(x, int))
    valid_ranges = df[target_columns].applymap(lambda x: x in [0, 1, 2])
    missing_values = df[target_columns].isnull().sum()


    print("Validation Results:")
    print(f"All values are integers? {'Yes' if valid_integers.all().all() else 'No'}")
    print(f"All values are within the valid range [0, 1, 2]? {'Yes' if valid_ranges.all().all() else 'No'}")
    print(f" Missing values:\n{missing_values}")
