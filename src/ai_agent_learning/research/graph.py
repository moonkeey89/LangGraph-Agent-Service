import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ai_agent_learning.research.graph_state import ResearchContext, ResearchState
from ai_agent_learning.research.workflow import (
    DEFAULT_RESEARCH_TOP_K,
    ResearchAnalysisAgent,
    ResearchAnalysisNode,
    ResearchCritic,
    ResearchEvidenceAgent,
    ResearchRetriever,
    ResearchRevision,
    ResearchSupervisor,
    ResearchSynthesizer,
    ValidateResearchBinding,
    finalize_research,
    route_after_analysis,
    route_after_critic,
    route_after_evidence,
    route_after_synthesis,
    route_after_validation,
    route_from_supervisor,
)
from ai_agent_learning.tools import calculate


logger = logging.getLogger(__name__)


def build_research_graph(
    llm: BaseChatModel | None = None,
    *,
    retriever: ResearchRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    calculate_tool: BaseTool = calculate,
    top_k: int = DEFAULT_RESEARCH_TOP_K,
    max_revisions: int = 1,
    supervisor_llm: BaseChatModel | None = None,
    analysis_llm: BaseChatModel | None = None,
    synthesis_llm: BaseChatModel | None = None,
    critic_llm: BaseChatModel | None = None,
    revision_llm: BaseChatModel | None = None,
    supervisor: ResearchSupervisor | None = None,
    evidence_agent: ResearchEvidenceAgent | None = None,
    analysis_agent: ResearchAnalysisNode | None = None,
    synthesizer: ResearchSynthesizer | None = None,
    critic: ResearchCritic | None = None,
    revision: ResearchRevision | None = None,
):
    """Compile an isolated, bounded ResearchFlow graph with no business writes."""
    supervisor_node = supervisor or ResearchSupervisor(
        _required_llm(supervisor_llm or llm, "ResearchSupervisor")
    )
    evidence_node = evidence_agent or ResearchEvidenceAgent(
        retriever,
        top_k=top_k,
    )
    analysis_node = analysis_agent or ResearchAnalysisAgent(
        _required_llm(analysis_llm or llm, "ResearchAnalysisAgent"),
        calculate_tool=calculate_tool,
    )
    synthesis_node = synthesizer or ResearchSynthesizer(
        _required_llm(synthesis_llm or llm, "ResearchSynthesizer")
    )
    critic_node = critic or ResearchCritic(
        _required_llm(critic_llm or llm, "ResearchCritic")
    )
    revision_node = revision or ResearchRevision(
        _required_llm(revision_llm or llm, "ResearchRevision")
    )
    validator = ValidateResearchBinding(max_revisions=max_revisions)

    graph = StateGraph(ResearchState, context_schema=ResearchContext)
    graph.add_node("research_validate_binding", validator.run)
    graph.add_node("research_supervisor", supervisor_node.run)
    graph.add_node("research_evidence_agent", evidence_node.run)
    graph.add_node("research_analysis_agent", analysis_node.run)
    graph.add_node("research_synthesize", synthesis_node.run)
    graph.add_node("research_critic", critic_node.run)
    graph.add_node("research_revise", revision_node.run)
    graph.add_node("research_finalize", finalize_research)

    graph.add_edge(START, "research_validate_binding")
    graph.add_conditional_edges(
        "research_validate_binding",
        route_after_validation,
        {
            "supervisor": "research_supervisor",
            "finalize": "research_finalize",
        },
    )
    graph.add_conditional_edges(
        "research_supervisor",
        route_from_supervisor,
        {
            "knowledge": "research_evidence_agent",
            "analysis": "research_analysis_agent",
            "synthesis": "research_evidence_agent",
            "direct": "research_synthesize",
        },
    )
    graph.add_conditional_edges(
        "research_evidence_agent",
        route_after_evidence,
        {
            "analysis": "research_analysis_agent",
            "synthesize": "research_synthesize",
            "finalize": "research_finalize",
        },
    )
    graph.add_conditional_edges(
        "research_analysis_agent",
        route_after_analysis,
        {
            "synthesize": "research_synthesize",
            "finalize": "research_finalize",
        },
    )
    graph.add_conditional_edges(
        "research_synthesize",
        route_after_synthesis,
        {
            "critic": "research_critic",
            "finalize": "research_finalize",
        },
    )
    graph.add_conditional_edges(
        "research_critic",
        route_after_critic,
        {
            "revise": "research_revise",
            "finalize": "research_finalize",
        },
    )
    graph.add_edge("research_revise", "research_finalize")
    graph.add_edge("research_finalize", END)

    logger.info("Compiled isolated ResearchFlow graph")
    return graph.compile(checkpointer=checkpointer)


def _required_llm(
    llm: BaseChatModel | None,
    component: str,
) -> BaseChatModel:
    if llm is None:
        raise ValueError(f"{component}缺少LLM或注入节点")
    return llm
