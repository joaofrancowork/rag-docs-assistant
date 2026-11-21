# Evaluation report

32 questions (including unanswerable ones) · embedder `fastembed/BAAI/bge-small-en-v1.5` · generator `extractive-baseline`

| Configuration | Hit@1 | Hit@5 | MRR | Answer acc. | Citation acc. | Abstention | Faithfulness | Relevance |
|---|---|---|---|---|---|---|---|---|
| dense | 86% | 100% | 0.92 | 62% | 83% | 33% | n/a | n/a |
| bm25 | 83% | 100% | 0.91 | 59% | 79% | 33% | n/a | n/a |
| hybrid | 100% | 100% | 1.00 | 62% | 83% | 33% | n/a | n/a |
| hybrid + lexical rerank | 100% | 100% | 1.00 | 62% | 83% | 33% | n/a | n/a |
| hybrid + LLM rerank | 83% | 100% | 0.91 | 59% | 79% | 33% | n/a | n/a |
