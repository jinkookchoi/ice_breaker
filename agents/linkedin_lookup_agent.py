import os
from loguru import logger

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily
from pydantic.v1.types import SecretStr

load_dotenv()

# Chapter 16. Linkedin Data Processing- Part 4: Custom Search Agent Implementation
# Note. langchain hub / stringPromptTemplate
# https://smith.langchain.com/hub/hwchase17/react
# ------
# Answer the following questions as best you can. You have access to the following tools:
#
# {tools}
#
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
#
# Begin!
#
# Question: {input}
#
# Thought:{agent_scratchpad}
# ------

def lookup(name: str) -> str:
    logger.info(name)
    llm = ChatOpenAI(
        temperature=0,
        # name="gpt-3.5-turbo",
        name="gpt-4",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                    Your answer should contain only a URL.
                    The URL does not contain 'pub' or 'dir' word.
                    The URL contains 'linkedin.com/in'"""


    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL.",
        )
    ]


    # Note. lanchain ReAct
    # https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/

    # ReAct: Synergizing Reasoning and Acting in Language Models
    # https://react-lm.github.io/

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True) # pyright: ignore

    logger.info("Invoke start ....")
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    linked_profile_url: str = result["output"]
    logger.info(linked_profile_url)
    return linked_profile_url
    

if __name__ == "__main__":
    lookup(name="Jinkook Choi")
    lookup(name="Eden Marco")
    lookup(name="Soojung Shin")
    lookup("Bill Gates")
    lookup("Jinkook Choi")
