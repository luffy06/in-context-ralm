from sentence_transformers import SentenceTransformer

def add_retriever_args(parser, retriever_type):
    if retriever_type == "sparse":
        parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
        parser.add_argument("--num_tokens_for_query", type=int, default=32)
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")

    elif retriever_type == "dense":
        parser.add_argument("--index_name", type=str, default="ivf65536-hnsw32-pq64")
        parser.add_argument("--num_tokens_for_query", type=int, default=32)
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")
        parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
        parser.add_argument("--retriever_dir", type=str, default="metadata/wikitext103-bert-base")
        parser.add_argument("--nprobe", type=int, default=512)
        parser.add_argument("--device_id", type=int, default=-1)
        parser.add_argument("--index_path", type=str, default=None)

    else:
        raise ValueError


def get_retriever(retriever_type, args, tokenizer):
    if retriever_type == "sparse":
        from ralm.retrievers.sparse_retrieval import SparseRetriever
        return SparseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            forbidden_titles_path=args.forbidden_titles_path,
            num_tokens_for_query=args.num_tokens_for_query,
        )
    elif retriever_type == "dense":
        from ralm.retrievers.dense_retrieval import DenseRetriever
        encoder = SentenceTransformer(args.encoder_name).to("cpu" if args.device_id == -1 else f"cuda:{args.device_id}")
        return DenseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            num_tokens_for_query=args.num_tokens_for_query,
            forbidden_titles_path=args.forbidden_titles_path,
            encoder=encoder,
            retriever_dir=args.retriever_dir,
            nprobe=args.nprobe,
            device_id=args.device_id,
            index_path=args.index_path,
        )
    raise ValueError
