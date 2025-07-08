import argparse
import os
import pandas as pd
from src.annotation import annotate_dataframe


def main():
    parser = argparse.ArgumentParser(description="LLM Annotation Experiment Runner")
    parser.add_argument('--data_path', type=str, required=True, help='CSV file to annotate')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g., llama3.1')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature setting')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--api_endpoint', type=str, required=True, help='API endpoint')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--runs', type=int, default=3, help='How many times to annotate each argument')
    parser.add_argument('--prompt_type', type=str, default="unbiased", choices=["unbiased", "bias"],
                        help='Type of prompt to use: unbiased or bias-injected')
    parser.add_argument('--bias_type', type=str, default="female", choices=["female", "male"],
                        help='If using bias-injected prompt, which bias: female or male')
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.data_path)
    # Drop NaN if needed
    df = df.dropna(subset=['argQ_id', 'claim_text', 'premise_texts', 'company_name'])

    # Annotate
    results_df = annotate_dataframe(
        df,
        args.model,
        args.api_endpoint,
        args.api_key,
        args.temperature,
        runs=args.runs,
        prompt_type=args.prompt_type,
        bias_type=args.bias_type
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"Annotation process completed and results saved to {args.output}")

if __name__ == "__main__":
    main()
