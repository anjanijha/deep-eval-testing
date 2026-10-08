# DeepEval LLM Evaluation Framework

A production-oriented framework for evaluating RAG applications using DeepEval.

## Architecture

```text
Golden Dataset
      ↓
Dataset Loader
      ↓
Pytest
      ↓
RAG Application
      ↓
LLM
      ↓
LLMTestCase
      ↓
DeepEval Metrics
      ↓
Quality Gate