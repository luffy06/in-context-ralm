DEVICE=1
CORPUS_SIZE='exact'
PROJECT_DIR=/disk3/xy/PROJECT/wsy/in-context-ralm
MODEL=opt-1.3b
MODEL_PATH=/disk3/xy/LM/$MODEL
OUTPUT_DIR=$PROJECT_DIR/outputs/$MODEL-$CORPUS_SIZE

if [[ -d $OUTPUT_DIR ]]; then
  rm -rf $OUTPUT_DIR
fi

if [[ ! -d $PROJECT_DIR/retrieval_files ]]; then
  mkdir $PROJECT_DIR/retrieval_files
fi

if [[ -f $PROJECT_DIR/retrieval_files/$MODEL-exact.txt ]]; then
  rm $PROJECT_DIR/retrieval_files/$MODEL-exact.txt
fi

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python prepare_retrieval_data.py \
    --retrieval_type exact \
    --tokenizer_name $MODEL_PATH \
    --max_length 1024 \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
    --stride 4 \
    --num_tokens_for_query 0 \
    --output_file retrieval_files/$MODEL-exact.txt \

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python eval_ppl.py \
    --model_name $MODEL_PATH \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 1024 \
    --retrieved_max_length 1024 \
    --retrieved_file retrieval_files/$MODEL-exact.txt

# CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
#   python prepare_retrieval_data.py \
#     --retrieval_type dense \
#     --tokenizer_name $MODEL_PATH \
#     --max_length 1024 \
#     --dataset_path wikitext \
#     --dataset_name wikitext-103-v1 \
#     --dataset_split test \
#     --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
#     --stride 4 \
#     --num_tokens_for_query 0 \
#     --num_docs 16 \
#     --output_file retrieval-$MODEL-$CORPUS_SIZE.txt \
#     --corpus_size $CORPUS_SIZE \
#     --encoder_name ~/LM/bert-base-uncased/ \
#     --retriever_dir metadata/wikipedia-split/ \
#     --nprobe 512 \
#     --device_id 0

# CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
#   python eval_lm.py \
#     --model_name $MODEL_PATH \
#     --dataset_path wikitext \
#     --dataset_name wikitext-103-v1 \
#     --dataset_split test \
#     --output_dir $OUTPUT_DIR \
#     --stride 4 \
#     --max_length 1024 \
#     --retrieved_file retrieval-$MODEL-$CORPUS_SIZE.txt

