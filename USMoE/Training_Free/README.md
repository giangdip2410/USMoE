# Unified Sparse Mixture of Experts



## Requirements

To run this code, you will need the following packages:

- `transformers`: Provides the necessary models and tokenizers for our experiments.
- `torch`: For running the models efficiently on your GPU or CPU.
- `bitsandbytes`: To support 4-bit quantization and reduce memory usage.

You can install all the dependencies with the following command:

```bash
pip install transformers torch bitsandbytes
```

## Usage

This code evaluates Mixture-of-Experts (MoE) models as embedding models on tasks from the Massive Text Embedding Benchmark (MTEB). You can specify the model, task type, and embedding method via command-line arguments.

### Command for Evaluation

To run the evaluation on OLMoE, use the following command:

```bash
bash run_exp.sh
```

### Argument Breakdown

- **`--base_model`**:  
    - Specify the base model to use for embedding extraction. Available options are:  
        - `"deepseek-ai/deepseek-moe-16b-base"`
        - `"Qwen/Qwen1.5-MoE-A2.7B"`
        - `"allenai/OLMoE-1B-7B-0924"`
  

- **`--task_types`**:  
    - Defines the type of tasks you want to run the evaluation on. Supported task types are:  
        - `'STS'`
        - `'Classification'`
        - `'Clustering'`
        - `'PairClassification'`
        - `'Reranking'`
        - `'Summarization'`

- **`--batch_size`**:  
    - Specifies the batch size to use during inference.

- **`--emb_info`**:  
    - Specifies the type of embeddings to evaluate. You have the following options:  
        - `'HS'`: Use the hidden state (HS) of the model as the embedding. This is the default strategy used in many pre-trained models.
        - `'RW'`: Use the routing weights (RW) from the Mixture of Experts (MoE) model as the embedding. RW is often more robust to prompt variations and captures high-level semantic information.

- **`--embed_method`**:  
    - Determines the prompting strategy to use when generating embeddings. Available options include:  
        - `'none'`: No specific prompting is applied. The embeddings are extracted directly from the model without any additional text input modifications.
        - `'prompteol'`: Use a prompt strategy where an end-of-line token or phrase is added to influence the model's generation of embeddings. This can help capture certain high-level features.




