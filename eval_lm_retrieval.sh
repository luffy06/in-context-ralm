DEVICE=0
CORPUS_SIZE='100K'
PROJECT_DIR=/disk3/xy/PROJECT/wsy/in-context-ralm
MODEL_PATH=/disk3/xy/LM/opt-2.7b
OUTPUT_DIR=$PROJECT_DIR/opt-2.7b-$CORPUS_SIZE

if [[ -d $OUTPUT_DIR ]]; then
  rm -rf $OUTPUT_DIR
fi

# CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
#   python prepare_retrieval_data.py \
#     --retrieval_type dense \
#     --tokenizer_name ~/LM/bert-base-uncased/ \
#     --max_length 1024 \
#     --dataset_path wikitext \
#     --dataset_name wikitext-103-v1 \
#     --dataset_split test \
#     --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
#     --stride 4 \
#     --num_tokens_for_query 32 \
#     --num_docs 16 \
#     --output_file retrieval-$CORPUS_SIZE.txt \
#     --corpus_size $CORPUS_SIZE \
#     --encoder_name ~/LM/bert-base-uncased/ \
#     --retriever_dir metadata/wikipedia-split/ \
#     --nprobe 512 \
#     --device_id 0

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python eval_lm.py \
    --model_name $MODEL_PATH \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 1024 \
    --retrieved_file retrieval-$CORPUS_SIZE.txt