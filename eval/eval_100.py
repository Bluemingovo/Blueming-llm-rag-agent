"""按题型统计 100 题评测结果（vector vs hybrid）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

from src.config import load_config
from src.embedding import build_embedder
from src.vector_store import VectorStore

cfg = load_config()
store = VectorStore(build_embedder(cfg.embedding_backend, model=cfg.embedding_model), cfg.vector_dir, cfg.docs_dir)
store.index()

qs = json.load(open("eval/questions_100.json", encoding="utf-8"))
TYPES = ["fact", "keyword", "multihop", "distractor"]


def is_hit(q, mode, k=4):
    got = store.retrieve(q["question"], k=k, mode=mode)
    names = {d.metadata.get("source", "").replace("\\", "/").split("/")[-1] for d in got}
    return q["source"] in names


for mode in ["vector", "hybrid"]:
    print(f"=== {mode.upper()} Hit@4 ===")
    total_hit = total = 0
    for t in TYPES:
        sub = [q for q in qs if q["type"] == t and q["source"]]
        h = sum(1 for q in sub if is_hit(q, mode))
        total_hit += h
        total += len(sub)
        print(f"  {t:12s}: {h}/{len(sub)} = {h / len(sub):.1%}")
    print(f"  {'TOTAL':12s}: {total_hit}/{total} = {total_hit / total:.1%}")
    print()
