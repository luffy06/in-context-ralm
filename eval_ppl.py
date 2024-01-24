import os
import argparse
import json
import pickle

import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer


def get_retrieved_doc_ids(retrieved_example, tokenizer):
    docs = [retrieval["text"] for retrieval in retrieved_example["retrieved_docs"]]
    docs = ' '.join(docs)
    doc_ids = tokenizer(docs, add_special_tokens=False, return_tensors="pt").input_ids
    return doc_ids

def eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length,
        model_max_length,
        output_dir=None,
        stride=4,
        normalization_level="word",
        retrieval_dataset=None,
        retrieval_max_length=256,
        ranking_strategy="first",
        num_docs_to_rank=1,
        num_tokens_to_rank_logprob=16
):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")

    max_input_length = max_length
    
    print("Max model input length:", model_max_length)
    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)

    nlls = []
    prev_end_loc = 0

    idx = 0
    all_token_ppls = []
    all_tokens_to_predict = []
    all_chosen_doc_ids = [None]
    num_inputs_no_retrieval = 0
    for begin_loc in tqdm(range(0, dataset_len, stride)):
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].clone()
        input_ids = input_ids.to(device)
        if retrieval_dataset is not None and len(retrieval_dataset[idx]["retrieved_docs"]) > 0:
            retrieved_example = retrieval_dataset[idx]
            retrieved_doc_ids = get_retrieved_doc_ids(retrieved_example, tokenizer).to(device)
            input_ids = torch.cat((retrieved_doc_ids, input_ids), 1).to(device)
            input_ids = input_ids[:, -model_max_length:]
            
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # Calculate per-token loss
            if trg_len < max_length:
                neg_log_likelihood = outputs.loss * trg_len
                lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                labels = target_ids[..., -trg_len:]
            else:
                neg_log_likelihood = outputs.loss * (max_length - 1)
                lm_logits = outputs.logits[..., :-1, :]
                labels = target_ids[..., 1:]
            neg_log_likelihood = neg_log_likelihood.to(torch.float32).squeeze().cpu()
            lm_logits = lm_logits.to(torch.float32)

            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).cpu()
            token_ppls = loss.tolist()
            tokens_to_predict = labels.view(-1).cpu().tolist()

        nlls.append(neg_log_likelihood)
        all_token_ppls.append(token_ppls)
        all_tokens_to_predict.append(tokens_to_predict)
        assert len(all_token_ppls) == len(all_tokens_to_predict)

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    assert retrieval_dataset is None or len(retrieval_dataset) == idx

    ppl = torch.exp(torch.stack(nlls).sum() / counter).item()
    print("Perplexity:", ppl)
    ppl_to_assert = np.exp(sum([sum(x) for x in all_token_ppls]) / counter)
    assert np.abs(ppl - ppl_to_assert) < 1e-3, f"{ppl:.3f}, {ppl_to_assert:.3f}"

    if output_dir is not None:
        d = {"eval_perplexity": ppl}
        if retrieval_dataset is not None:
            d["num_input_no_retrieval"] = num_inputs_no_retrieval
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    print_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)

    eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length=max_length,
        model_max_length=model_max_length,
        output_dir=args.output_dir,
        stride=args.stride,
        normalization_level=args.normalization_level,
        retrieval_dataset=retrieval_dataset,
        retrieval_max_length=args.retrieved_max_length,
        ranking_strategy=args.ranking_strategy,
        num_docs_to_rank=args.num_docs_to_rank,
        num_tokens_to_rank_logprob=args.ranking_logprob_past_tokens,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)
    parser.add_argument("--retrieved_max_length", type=int, default=256)
    parser.add_argument("--ranking_strategy", type=str, choices=["first", "logprob", "oracle", "random"], default="first")
    parser.add_argument("--num_docs_to_rank", type=int, default=-1)
    parser.add_argument("--ranking_logprob_past_tokens", type=int, default=16)

    args = parser.parse_args()

    main(args)
