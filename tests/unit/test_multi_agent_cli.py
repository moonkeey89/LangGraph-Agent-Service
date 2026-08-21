import unittest
from unittest.mock import Mock, patch

from ai_agent_learning.multi_agent_cli import create_supervisor_app


class MultiAgentCliTests(unittest.TestCase):
    @patch("ai_agent_learning.multi_agent_cli.build_supervisor_graph")
    @patch("ai_agent_learning.multi_agent_cli.create_llm")
    def test_factory_reuses_config_checkpointer_and_store(
        self,
        create_llm,
        build_supervisor_graph,
    ):
        settings = Mock()
        settings.memory_manager_confidence_threshold = 0.8
        settings.supervisor_max_subagent_calls = 5
        checkpointer = Mock()
        store = Mock()

        create_supervisor_app(settings, checkpointer, store)

        build_supervisor_graph.assert_called_once_with(
            create_llm.return_value,
            checkpointer=checkpointer,
            store=store,
            memory_confidence_threshold=0.8,
            max_subagent_calls=5,
        )


if __name__ == "__main__":
    unittest.main()
