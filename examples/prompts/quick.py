# https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/

from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from loguru import logger

logger.info("### PromptTemplate ###")
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke aboutn {content}."
)
res = prompt_template.format(adjective="funny", content="chickens")
logger.info(res)

logger.info("### ChatPromptTemplate ###")
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)
messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
logger.info(messages)
for message in messages:
    logger.info(message)

logger.info("### ChatPromptTemplate, SystemMessage, HumanMessagePromptTemplate ###")
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
        HumanMessage(content=("Hello")),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
logger.info(messages)
for message in messages:
    logger.info(message)

logger.info("### Message Prompts ###")
prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
message = chat_message_prompt.format(subject="force")
logger.info(message)

logger.info("### MessagePlaceholder ###")
human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)
logger.info(chat_prompt)


human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

messages = chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()
logger.info(messages)

logger.info("### LCEL ###")
