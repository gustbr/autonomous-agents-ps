from tools.search_vector import search_vector
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Union
from pydantic import BaseModel, Field

from prompts import replanner_prompt, section_writer_instructions

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = hub.pull("ih/ih-react-agent-executor")

tools = [search_vector]


agent_executor = create_react_agent(llm, tools, state_modifier=prompt)


class PlanExecute(TypedDict):
    input: str
    context: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    recursion_limit: int


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )



replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)


from typing import Literal
from langgraph.graph import END


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def user_context(state: PlanExecute):
    result = search_vector(state["input"])
    return {"context": result}

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"] + "\n\n" + state["context"])]})
    return {"plan": plan.steps, "recursion_limit": len(plan.steps)}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if len(state["past_steps"]) - 1 >= state["recursion_limit"] or "response" in state and state["response"]:
        print("SHOULD END")
        return 'write_section'
    else:
        return "agent"

def write_section(state: PlanExecute):
    print("STATE:", state)
    system_message = section_writer_instructions.format(input=state["input"])
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {state['past_steps']}")])
    save_section_content(section)

def save_section_content(section, filename="output_section.md"):
    # Extract just the content from the section
    content = section.content if hasattr(section, 'content') else str(section)

    # Write the content to a file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Section has been saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")




workflow = StateGraph(PlanExecute)

# Add the context node
workflow.add_node("user_context", user_context)
# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_node("write_section", write_section)

workflow.add_edge(START, "user_context")
workflow.add_edge("user_context", "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", "write_section"],
)
workflow.add_edge("write_section", END)


# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# Instead of graph.draw_mermaid_png("workflow_graph.png")
graph = app.get_graph(xray=True)
png_data = graph.draw_mermaid_png()  # Get the PNG data

# Save the PNG data to a file
with open("workflow_graph.png", "wb") as f:
    f.write(png_data)


config = {"recursion_limit": 10}
inputs = {"input": "How can a person, when faced with adversity or a stressful problem, manage to focus solely on the facts and not allow themselves to be overwhelmed by negative or irrational thoughts that increase stress?"}
async def main():
    async for event in app.astream(inputs):
        if 'planner' in event:
            num_steps = len(event['planner']['plan'])
            print("PLAN",event['planner'])
            print(f"Number of steps in plan: {num_steps}")

import asyncio
asyncio.run(main())
