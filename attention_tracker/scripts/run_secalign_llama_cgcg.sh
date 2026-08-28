MODELS=("secalign_llama3_8b-attn")

for MODEL in "${MODELS[@]}"; do
    # Run as a module so relative imports (e.g. `from .utils`) inside
    # `attention_tracker/run_dataset.py` work correctly.
    python -m attention_tracker.run_dataset \
                    --model_name ${MODEL} \
                    --seed 0 \
                    --results_folder "results_folder" \
                    --attack_name "cgcg" \
                    --important_heads "subset_llm_30_n4"

done
