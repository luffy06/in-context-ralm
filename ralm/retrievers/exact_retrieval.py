import json
import multiprocessing

from ralm.retrievers.base_retrieval import BaseRetriever


class ExactRetriever(BaseRetriever):
    def __init__(self, tokenizer, num_tokens_for_query, forbidden_titles_path):
        super(ExactRetriever, self).__init__(tokenizer=tokenizer)
        self.num_tokens_for_query = num_tokens_for_query

    def _get_query_string(self, sequence_input_ids, begin_location, target_begin_location, end_location, title=None):
        # We isolate the prefix to make sure that we don't take tokens from the future:
        query_tokens = sequence_input_ids[0, begin_location:target_begin_location]
        if self.num_tokens_for_query != 0:
          query_tokens = query_tokens[-self.num_tokens_for_query:]
        query_str = self.tokenizer.decode(query_tokens)
        return query_str

    def retrieve(self, sequence_input_ids, dataset, k=1):
        for i, d in enumerate(dataset):
            query = self._get_query_string(
                        sequence_input_ids,
                        d["begin_location"],
                        d["target_begin_location"],
                        d["end_location"],
                        d["title"] if "title" in d else None
                    )
            dataset[i]["query"] = query
            dataset[i]["retrieved_docs"] = [{"text": self.tokenizer.decode(sequence_input_ids[0, d["begin_location"]:d["end_location"]]), "score": 1}]
        return dataset
