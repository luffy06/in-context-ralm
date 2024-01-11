DEVICE=3
CORPUS_SIZE='1M'
PROJECT_DIR=/disk3/xy/PROJECT/wsy/in-context-ralm
MODEL=opt-2.7b
MODEL_PATH=/disk3/xy/LM/$MODEL
OUTPUT_DIR=$PROJECT_DIR/$MODEL-$CORPUS_SIZE

if [[ -d $OUTPUT_DIR ]]; then
  rm -rf $OUTPUT_DIR
fi

if [[ -f $PROJECT_DIR/retrieval-$MODEL-$CORPUS_SIZE.txt ]]; then
  rm $PROJECT_DIR/retrieval-$MODEL-$CORPUS_SIZE.txt
fi

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python prepare_retrieval_data.py \
    --retrieval_type dense \
    --tokenizer_name $MODEL_PATH \
    --max_length 1024 \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
    --stride 4 \
    --num_tokens_for_query 32 \
    --num_docs 16 \
    --output_file retrieval-$MODEL-$CORPUS_SIZE.txt \
    --corpus_size $CORPUS_SIZE \
    --encoder_name ~/LM/bert-base-uncased/ \
    --retriever_dir metadata/wikipedia-split/ \
    --nprobe 512 \
    --device_id 0

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 \
  python eval_lm.py \
    --model_name $MODEL_PATH \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 1024 \
    --retrieved_file retrieval-$MODEL-$CORPUS_SIZE.txt