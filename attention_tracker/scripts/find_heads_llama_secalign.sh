MODELS=("secalign_llama3_8b-attn")
for MODEL in "${MODELS[@]}"; do
    python3 -m attention_tracker.select_head \
                            --model_name ${MODEL} \
                            --num_data 30 \
                            --dataset llm  >> "analysis_llm.txt"
done
