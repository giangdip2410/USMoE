base_models=(
    "allenai/OLMoE-1B-7B-0924"
)

task_types=(
    "Classification"
    "STS"
    "Clustering"
    "PairClassification"
    "Reranking"
    "Summarization"
)

emb_infos=(
    "HS"
)

embed_methods=(
    "none"
    "prompteol"
)

# Loop through each combination of parameters
for base_model in "${base_models[@]}"; do
    for task_type in "${task_types[@]}"; do
        for emb_info in "${emb_infos[@]}"; do
            for embed_method in "${embed_methods[@]}"; do
                echo "Running evaluation with: Base Model=$base_model, Task Type=$task_type, Emb Info=$emb_info , Emb Method=$embed_method"
                python eval_mteb.py \
                    --base_model "$base_model" \
                    --use_4bit \
                    --task_types "$task_type" \
                    --batch_size 128 \
                    --emb_info "$emb_info" \
                    --embed_method "$embed_method"
            done
        done
    done
done