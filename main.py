from dotenv import load_dotenv
from langchain.agents import tool as agent_tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_log_to_str
from typing import Union, List

from callbacks import AgentCallbackHandler


load_dotenv()

@agent_tool
def get_text_length(text:str) -> int:
    """Returns the length of the text"""
    return len(text)

@agent_tool
def write_haiku(topic: str) -> str:
    """Writes a haiku about the topic given."""
    return ChatOpenAI().predict(text=f"Write a haiku about {topic}")

def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found")

if __name__ == '__main__':
    tools = [get_text_length, write_haiku]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join([tool.name for tool in tools]))
    
    stop_on_tokens = ["Observation:"]

    llm = ChatOpenAI(temperature=0, stop=stop_on_tokens, callbacks=[AgentCallbackHandler()])
    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
        )
    
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "Write a haiku about dogs and then count the length by characters.",
                "agent_scratchpad": intermediate_steps,
            })
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, str(observation)))

    print(agent_step.return_values)
