"""LangGraph foundation for agentic SEC research."""

from langgraph.graph import StateGraph, START, END

from evidenceroute.research.llm_client import LLMClient
from evidenceroute.research.budget import BudgetExceededError
from evidenceroute.research.agentic.state import AgentState
from evidenceroute.research.agentic.planner import (
    create_research_plan,
)
from evidenceroute.research.agentic.tools import (
    SECTools,
    SearchRequest,
)


def build_agent_graph(
    tools: SECTools,
    llm_client: LLMClient,
    top_k: int = 5,
):

    def planner_node(state: AgentState) -> dict:
        try:
            plan = create_research_plan(
                state["question"],
                llm_client,
            )

            return {
                "plan": plan,
                "status": "planned",
                "trace": [{
                    "node": "planner",
                    "task_count": len(plan.tasks),
                }],
            }

        except BudgetExceededError:
            return {
                "status": "budget_exhausted",
                "errors": ["Planning budget exhausted"],
            }

    def research_node(state: AgentState) -> dict:
        evidence = {}
        completed = []
        errors = list(state.get("errors", []))
        trace = list(state.get("trace", []))

        for index, task in enumerate(state["plan"].tasks):

            try:
                request = SearchRequest(
                    query=task.query,
                    top_k=top_k,
                    tickers=task.tickers,
                    fiscal_years=task.fiscal_years,
                    section_ids=task.section_ids,
                )

                results = tools.search_chunks(request)

                for result in results:
                    chunk_id = result.chunk.chunk_id

                    if chunk_id not in evidence:
                        evidence[chunk_id] = result

                completed.append(index)

                trace.append({
                    "node": "research",
                    "task_index": index,
                    "query": task.query,
                    "result_count": len(results),
                })

            except Exception as exc:
                errors.append(
                    f"Task {index}: {type(exc).__name__}"
                )

        return {
            "evidence": list(evidence.values()),
            "completed_tasks": completed,
            "errors": errors,
            "trace": trace,
            "iteration": 1,
            "status": (
                "researched" if evidence else "no_evidence"
            ),
        }

    def route_after_planning(state: AgentState) -> str:
        if state.get("status") == "budget_exhausted":
            return "stop"

        return "research"

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "research": "research",
            "stop": END,
        },
    )

    graph.add_edge("research", END)

    return graph.compile()
