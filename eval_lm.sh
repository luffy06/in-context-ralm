DEVICE=2
PROJECT_DIR=/disk3/xy/PROJECT/wsy/in-context-ralm
MODEL=opt-6.7b # llama-7b llama-13b opt-1.3b opt-2.7b opt-6.7b
MODEL_PATH=/disk3/xy/LM/$MODEL
OUTPUT_DIR=$PROJECT_DIR/outputs/$MODEL-no-retrieval

if [[ -d $OUTPUT_DIR ]]; then
  rm -rf $OUTPUT_DIR
fi

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python eval_lm.py \
    --model_name $MODEL_PATH \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 1024
