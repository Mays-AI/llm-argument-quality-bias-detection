# LLM-based Argument Quality Annotation & Bias Detection


This project explores the annotation of financial arguments for quality and bias detection via large language models (LLMs).
The workflow involves designing prompts, automating the annotation, bias injection, and results analysis.
We utilize a publicly available dataset of [FinArgQuality](https://github.com/Alaa-Ah/The-FinArgQuality-dataset-Quality-of-managers-arguments-in-Eearnings-Conference-Calls).

 

### Requirements

1. Install the required dependencies: 

```sh
pip install -r requirements.txt
```


#### How to Reproduce Results

2. Prepare your data: Store CSV input files in the data/ folder.


3. Run the annotation script via `run_annotation.py`, ensuring the data path is set as an environment variable.
 
```sh
  python scripts/run_annotation.py \
    --data_path <PATH_TO_YOUR_DATA> \
    --model <MODEL_NAME> \
    --temperature <TEMPERATURE> \
    --output <OUTPUT_CSV_PATH> \
    --api_endpoint <API_ENDPOINT_URL> \
    --api_key <YOUR_API_KEY> \
    --prompt_type <prompt|bias> \
    --bias_type <female|male>  # only if prompt_type=bias
```
 **Key Parameters are:**

- --data_path: Path to the input CSV data file
- --model: Model name (e.g., llama3.1)
- --temperature: Sampling temperature for the model
- --output: Output path for the annotated results
- --api_endpoint: API endpoint for LLM inference
- --api_key: API key for authentication
- --prompt_type: Type of prompt (basic or with bias)
-  - If --prompt_type bias, specify --bias_type as either female or male.
  



### Jupyter Notebook: 
Open notebooks/.ipynb to run end-to-end experiments, visualizations, and metrics.


## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Scikit-learn](https://scikit-learn.org/)
- [Statsmodels](https://www.statsmodels.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

