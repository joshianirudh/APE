
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="LOFT-HF")

    ## model
    parser.add_argument('--model-id', type=str, default='', help='model id')

    ## dataset path
    parser.add_argument('--data-folder', type=str, default='', help='path to the datafolder of ChatRAG Bench')
    parser.add_argument('--output-folder', type=str, default='', help='path to the datafolder of ChatRAG Bench')
    parser.add_argument('--eval-dataset', type=str, default='')
    parser.add_argument('--nq-query-path', type=str, default='nq/1m/test_queries.jsonl')
    parser.add_argument('--nq-context-path', type=str, default='nq/1m/corpus.jsonl')
    parser.add_argument('--arguana-query-path', type=str, default='arguana/128k/test_queries.jsonl')
    parser.add_argument('--arguana-context-path', type=str, default='arguana/128k/corpus.jsonl')
    parser.add_argument('--fever-query-path', type=str, default='fever/128k/test_queries.jsonl')
    parser.add_argument('--fever-context-path', type=str, default='fever/128k/corpus.jsonl')
    parser.add_argument('--fiqa-query-path', type=str, default='fiqa/128k/test_queries.jsonl')
    parser.add_argument('--fiqa-context-path', type=str, default='fiqa/128k/corpus.jsonl')
    parser.add_argument('--msmacro-query-path', type=str, default='msmacro/128k/test_queries.jsonl')
    parser.add_argument('--msmacro-context-path', type=str, default='msmacro/128k/corpus.jsonl')
    parser.add_argument('--quora-query-path', type=str, default='quora/128k/test_queries.jsonl')
    parser.add_argument('--quora-context-path', type=str, default='quora/128k/corpus.jsonl')
    parser.add_argument('--scifact-query-path', type=str, default='scifact/128k/test_queries.jsonl')
    parser.add_argument('--scifact-context-path', type=str, default='scifact/128k/corpus.jsonl')

    ## others
    parser.add_argument('--out-seq-len', type=int, default=64)
    parser.add_argument('--num-ctx', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--max-lengths', type=int, default=8192)

    args = parser.parse_args()

    return args
