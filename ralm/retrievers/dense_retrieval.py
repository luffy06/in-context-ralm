import json
import os
import lmdb
import faiss
import pickle
import multiprocessing
import math
import logging
import numpy as np
logger = logging.getLogger(__name__)
from glob import glob
from ralm.retrievers.base_retrieval import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(
        self, 
        tokenizer, 
        index_name, 
        num_tokens_for_query, 
        forbidden_titles_path, 
        encoder, 
        retriever_dir, 
        corpus_size,
        nprobe, 
        device_id=-1, 
        index_path=None
    ):
        super(DenseRetriever, self).__init__(tokenizer=tokenizer)
        self._set_searcher(index_name, encoder, retriever_dir, corpus_size, nprobe, device_id, index_path)
        self.num_tokens_for_query = num_tokens_for_query

        self.forbidden_titles = self._get_forbidden_titles(forbidden_titles_path)

    def _set_searcher(
        self, 
        index_name, 
        encoder, 
        retriever_dir, 
        corpus_size,
        nprobe, 
        device_id=-1, 
        index_path=None
    ):
        with open(os.path.join(retriever_dir, "metadata.json"), "r") as fin:
            self.metadata = json.load(fin)
        db_path = self.metadata["db_path"] if "db_path" in self.metadata else os.path.join(retriever_dir, os.path.join("db", corpus_size))
        if index_path == None:
            if "index_path" in self.metadata:
                index_path = self.metadata["index_path"]
            else:
                index_dir = os.path.join(retriever_dir, os.path.join("index", corpus_size))
                index_path_list = glob(os.path.join(index_dir, "*.index"))
                if index_name != None:
                    for index_path_i in index_path_list:
                        name = index_path_i.split("/")[-1].split(".")[0]
                        if index_name == name:
                            index_path = index_path_i
                            break
                else:
                    index_path = index_path_list[0]
                assert index_path != None
        
        self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        self.device_id = device_id
        if device_id >= 0:
            gpu_res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(gpu_res, device_id, self.index, co)
        self.index.nprobe = nprobe
        self.env = lmdb.open(db_path, readonly=True, readahead=True)
        self.retrieval_dim = self.metadata["emb_dim"]
        self.encoder = encoder
    
    def __del__(self):
        self.env.close()

    def _get_forbidden_titles(self, forbidden_titles_path):
        if forbidden_titles_path is None:
            return []
        with open(forbidden_titles_path, "r") as f:
            forbidden_titles = [line.strip() for line in f]
        return set(forbidden_titles)

    def _get_title_from_retrieved_document(self, doc):
        title, _ = doc.split("\n")
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        return title

    def _retrieve_no_forbidden(self, query_str):
        k = 16
        prev_k = 0
        while True:
            retrieved_res = self.searcher.search(query_str, k=k)
            for idx in range(prev_k, k):
                res_dict = json.loads(retrieved_res[idx].raw)
                context_str = res_dict["contents"]
                title = self._get_title_from_retrieved_document(context_str)
                if title not in self.forbidden_titles:
                    return context_str
            prev_k = k
            k *= 2

    def _get_query_string(self, sequence_input_ids, begin_location, target_begin_location, end_location, title=None):
        # We isolate the prefix to make sure that we don't take tokens from the future:
        query_tokens = sequence_input_ids[0, begin_location:target_begin_location]
        if self.num_tokens_for_query > 0:
            query_tokens = query_tokens[-self.num_tokens_for_query:]
        query_str = self.tokenizer.decode(query_tokens)
        return query_str

    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = [
            self._get_query_string(
                sequence_input_ids,
                d["begin_location"],
                d["target_begin_location"],
                d["end_location"],
                d["title"] if "title" in d else None
            )
            for d in dataset
        ]
        assert len(queries) == len(dataset)

        all_res = self.batch_search(
            queries,
            qids=[str(i) for i in range(len(queries))],
            k=max(100, 4*k) if self.forbidden_titles else k,
            threads=multiprocessing.cpu_count()
        )

        for qid, res in all_res.items():
            qid = int(qid)
            d = dataset[qid]
            d["query"] = queries[qid]
            d["retrieved_docs"] = res
        return dataset


    def batch_search(self, queries, qids, k, threads):
        query_embs = self.encoder.encode(queries)
        distances, ids = self.index.search(query_embs, k)
        txn = self.env.begin()
        all_contexts = {}
        for query_i in range(query_embs.shape[0]):
            neighbor_text = []
            qid = qids[query_i]
            for j in range(k):
                score = str(-distances[query_i][j])
                if ids[query_i][j] != -1:
                    key = txn.get(str(ids[query_i][j]).encode())
                    assert key != None, f"Cannot find key {ids[query_i][j]}"
                    value = pickle.loads(key)
                    text = str(value["text"])
                    neighbor_text.append({"text": text, "score": score})
            all_contexts[qid] = neighbor_text
        return all_contexts
