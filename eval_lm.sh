DEVICE=0
PROJECT_DIR=/disk3/xy/PROJECT/wsy/in-context-ralm
MODEL_PATH=/disk3/xy/LM/opt-1.3b
OUTPUT_DIR=$PROJECT_DIR/opt-1.3b-no-retrieval

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
