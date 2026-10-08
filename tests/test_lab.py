import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class LabChecks(unittest.TestCase):
    def test_rag_chain_and_contexts(self):
        rag = importlib.import_module("01_langsmith_rag_pipeline")
        evaluation = importlib.import_module("03_ragas_evaluation")
        embeddings = DeterministicFakeEmbedding(size=16)
        llm = FakeListChatModel(responses=["Câu trả lời thử", "Câu trả lời thử"])
        with patch.object(rag, "get_embeddings", return_value=embeddings), patch.object(rag, "get_llm", return_value=llm):
            store = rag.setup_vectorstore()
            chain, retriever = rag.build_rag_chain(store)
            self.assertEqual(rag.ask(chain, "What is RAG?"), "Câu trả lời thử")
        result = evaluation.run_rag(retriever, llm, evaluation.PROMPT_V1, "What is RAG?")
        self.assertEqual(result["answer"], "Câu trả lời thử")
        self.assertEqual(len(result["contexts"]), 3)
        self.assertTrue(all(isinstance(context, str) for context in result["contexts"]))

    def test_routing_and_prompts(self):
        ab = importlib.import_module("02_prompt_hub_ab_routing")
        versions = [ab.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(versions, [ab.get_prompt_version(f"req-{i:04d}") for i in range(50)])
        self.assertEqual(set(versions), {ab.PROMPT_V1_NAME, ab.PROMPT_V2_NAME})
        self.assertEqual(set(ab.PROMPT_V1.input_variables), {"context", "question"})
        self.assertEqual(set(ab.PROMPT_V2.input_variables), {"context", "question"})

    def test_ragas_dataset(self):
        evaluation = importlib.import_module("03_ragas_evaluation")
        row = {"question": "q", "answer": "a", "contexts": ["one", "two"], "reference": "r"}
        sample = evaluation.build_ragas_dataset([row]).samples[0]
        self.assertEqual(sample.retrieved_contexts, ["one", "two"])
        self.assertEqual(sample.reference, "r")
        metrics = {name: [0.8, 1.0] for name in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")}
        with patch.object(evaluation, "get_llm"), patch.object(evaluation, "get_embeddings"), patch.object(evaluation, "evaluate", return_value=metrics):
            self.assertEqual(evaluation.run_ragas_eval([row], "v1")["faithfulness"], 0.9)

    def test_guardrails_fix(self):
        guardrails = importlib.import_module("04_guardrails_validator")
        pii_guard = guardrails.Guard().use(guardrails.PIIDetector(on_fail=guardrails.OnFailAction.FIX))
        self.assertEqual(pii_guard.validate("Call (555) 867-5309").validated_output, "Call [PHONE_REDACTED]")
        self.assertEqual(pii_guard.validate("Clean text").validated_output, "Clean text")
        json_guard = guardrails.Guard().use(guardrails.JSONFormatter(on_fail=guardrails.OnFailAction.FIX))
        self.assertEqual(json.loads(json_guard.validate("{'a': 1,}").validated_output), {"a": 1})
        self.assertIn("error", json.loads(json_guard.validate("broken").validated_output))


if __name__ == "__main__":
    unittest.main()
