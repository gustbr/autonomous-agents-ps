from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.runnables.config import RunnableConfig

ROUTER_SYSTEM_MESSAGE = """Read the user profil:

{memory}

Is there any information missing ? Answer with "yes" or "no"."""

# System messages
MODEL_SYSTEM_MESSAGE = """You are an assistant helping to understand the user better.
You need to collect: their name, their mindset improvement goals, and their current challenges.

Current user information:
{memory}

If any of this information is missing, ask ONE question to collect it.
If you have all required information, you can engage in normal conversation."""

CREATE_MEMORY_INSTRUCTION = """Review the conversation and update user information.
Current information:
{memory}

Required information:
- Username/Name
- Goals/Focus Areas (what they want to achieve)
- Current Challenges

Instructions:
1. Extract new information from the conversation
2. Update or add to existing information
3. Format as a clear bullet list
4. Note any missing required information

Keep only factual information stated by the user."""

def print_last_human_message(messages):
    # Filter only HumanMessages and get the last one
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if human_messages:
        last_human_message = human_messages[-1]
        print("Last Human Message:", last_human_message.content)


class ChatBot:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.across_thread_memory = InMemoryStore()
        self.within_thread_memory = MemorySaver()
        self.builder = self._create_graph()
        self.graph = self.builder.compile(
            checkpointer=self.within_thread_memory,
            store=self.across_thread_memory
        )

    def _create_graph(self):
        # Move all the graph creation logic here
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", self.call_model)
        builder.add_node("write_memory", self.write_memory)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "write_memory")
        builder.add_conditional_edges(
            "write_memory",
            self.route_tools,
            {"call_model": "call_model", END: END},
        )
        builder.add_edge("write_memory", END)
        return builder

    def route_tools(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        # Same as your existing route_tools but as instance method
        print("Routing to call_model")
        user_id = config["configurable"]["user_id"]
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")
        memory_content = existing_memory.value.get('memory') if existing_memory else "No information yet."

        system_msg = ROUTER_SYSTEM_MESSAGE.format(memory=memory_content)
        response = self.model.invoke([SystemMessage(content=system_msg)])
        print(response.content)
        if response.content == "Yes":
            return "call_model"
        else:
            return END

    def call_model(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        # Same as your existing call_model but as instance method
        print("Calling model")
        """Generate response based on memory and current conversation."""

        user_id = config["configurable"]["user_id"]
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")

        memory_content = existing_memory.value.get('memory') if existing_memory else "No information yet."
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=memory_content)
        response = self.model.invoke([SystemMessage(content=system_msg)] + state["messages"])
        # Print the assistant's response
        print("Assistant:", response.content)

        # Get user input
        user_input = input("> ")
        state["messages"].append(HumanMessage(content=user_input))

        return {"messages": response}

    def write_memory(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        # Same as your existing write_memory but as instance method
        print("Writing memory")
        #user_input = print_last_human_message(state["messages"])
        """Update memory with new information from conversation."""

        user_id = config["configurable"]["user_id"]
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")

        memory_content = existing_memory.value.get('memory') if existing_memory else "No information yet."

        system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=memory_content)
        new_memory = self.model.invoke([SystemMessage(content=system_msg)] + state["messages"])
        store.put(namespace, "user_memory", {"memory": new_memory.content})

        return state

    def chat(self, user_input: str = None):
        if user_input is None:
            user_input = input("Hello! I'm here to learn more about you. What's your name? > ")
        return self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": "1", "user_id": "1"}}
        )

# Keep this for direct script execution
if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.chat()
