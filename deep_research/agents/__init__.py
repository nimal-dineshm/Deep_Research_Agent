"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.prompt import final_report_generation_prompt
from deep_research.state import AgentState, AgentInputState
from deep_research.agents.scoping_agent import clarify_with_user, write_research_brief
from deep_research.agents.supervisor_agent import supervisor_agent

_DEFAULT_FINAL_REPORT_MODEL = "anthropic:claude-haiku-4-5-20251001"

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """
    Final report generation node.
    
    Synthesizes all research findings into a comprehensive final report
    """
    
    model_name = config.get("configurable", {}).get("final_report_model", _DEFAULT_FINAL_REPORT_MODEL)
    writer_model = init_chat_model(model_name, temperature=0.7)

    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    
    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()