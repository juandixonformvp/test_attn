from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from attention_graph.common import HttpDriver, connect, gini, js_divergence, normalized_entropy


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class MetricTests(unittest.TestCase):
    def test_distribution_metrics_capture_balance_and_separation(self):
        self.assertAlmostEqual(normalized_entropy([10, 10]), 1.0)
        self.assertEqual(normalized_entropy([20, 0]), 0.0)
        self.assertAlmostEqual(js_divergence({"praise": 10}, {"attack": 10}), 1.0)
        self.assertAlmostEqual(gini([1, 1, 1]), 0.0)

    def test_cascade_metrics_separate_chain_from_star(self):
        module = load_script("05_conversation_cascades.py")
        chain = [
            {"comment_id": "a", "parent_id": None, "account": "u1"},
            {"comment_id": "b", "parent_id": "a", "account": "u2"},
            {"comment_id": "c", "parent_id": "b", "account": "u3"},
            {"comment_id": "d", "parent_id": "c", "account": "u4"},
            {"comment_id": "e", "parent_id": "d", "account": "u5"},
        ]
        star = [{"comment_id": str(i), "parent_id": None, "account": f"u{i}"} for i in range(5)]
        chain_metrics = module.cascade_metrics(chain)
        star_metrics = module.cascade_metrics(star)
        self.assertGreater(chain_metrics["max_depth"], star_metrics["max_depth"])
        self.assertGreater(chain_metrics["structural_virality"], star_metrics["structural_virality"])

    def test_weighted_hits_rewards_broad_amplification(self):
        module = load_script("09_amplifier_analysis.py")
        edges = [
            {"account": "broad", "claim": "c1", "expressions": 2},
            {"account": "broad", "claim": "c2", "expressions": 2},
            {"account": "narrow", "claim": "c1", "expressions": 1},
        ]
        hubs, authorities = module.weighted_hits(edges)
        self.assertGreater(hubs["broad"], hubs["narrow"])
        self.assertGreater(authorities["c1"], authorities["c2"])

    def test_http_driver_maps_transaction_rows(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [{"columns": ["name", "score"], "data": [{"row": ["claim", 3]}]}],
            "errors": [],
        }
        driver = HttpDriver("https://graph.example", "neo4j", "secret", "neo4j")
        driver.session.post = Mock(return_value=response)
        self.assertEqual(driver.query("RETURN 1", {}, "neo4j"), [{"name": "claim", "score": 3}])
        called_url = driver.session.post.call_args.args[0]
        self.assertEqual(called_url, "https://graph.example/db/neo4j/tx/commit")
        driver.close()

    def test_plain_http_requires_explicit_opt_in(self):
        args = SimpleNamespace(uri="http://graph.example", user="neo4j", database="neo4j")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                connect(args)


if __name__ == "__main__":
    unittest.main()
