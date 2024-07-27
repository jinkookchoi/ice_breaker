# from typing import Tuple

# from langchain.prompts.prompt import PromptTemplate
# from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
# from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
# from chains.custom_chains import (
#     get_summary_chain,
#     get_interests_chain,
#     get_ice_breaker_chain,
# )
# from third_parties.linkedin import scrape_linkedin_profile
# from third_parties.twitter import scrape_user_tweets, scrape_user_tweets_mock
# from output_parsers import (
#     Summary,
#     IceBreaker,
#     TopicOfInterest,
# )


# def ice_break_with(
# name: str,
# ) -> Tuple[Summary, TopicOfInterest, IceBreaker, str]:
# linkedin_username = linkedin_lookup_agent(name=name)
# linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
#
# twitter_username = twitter_lookup_agent(name=name)
# tweets = scrape_user_tweets(username=twitter_username)

# summary_chain = get_summary_chain()
# summary_and_facts: Summary = summary_chain.invoke(
#     input={"information": linkedin_data, "twitter_posts": tweets},
# )
#
# interests_chain = get_interests_chain()
# interests: TopicOfInterest = interests_chain.invoke(
#     input={"information": linkedin_data, "twitter_posts": tweets}
# )
#
# ice_breaker_chain = get_ice_breaker_chain()
# ice_breakers: IceBreaker = ice_breaker_chain.invoke(
#     input={"information": linkedin_data, "twitter_posts": tweets}
# )
#
# return (
#     summary_and_facts,
#     interests,
#     ice_breakers,
#     linkedin_data.get("profile_pic_url"),
# )
# if __name__ == "__main__":
# pass


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#
# Chapter 8. Your First LangChain application - Chaining a simple promp
# Chapter 9. Using Open Source Models With LangChain (Ollama, Llama3, Mistral)
#

from loguru import logger
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets_mock

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=False)

    # Note. Proxycurl credits required!!!
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/jinkookchoi/", mock=False)

    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/jinkookchoi/", mock=True)
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets_mock(username=twitter_username)

    # mock_information = """
    #     Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in space company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI. He is one of the wealthiest people in the world; as of July 2024, Forbes estimates his net worth to be US$221 billion.[4]
    #     Musk was born in Pretoria to model Maye and businessman and engineer Errol Musk, and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In October 2002, eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.
    #     In 2004, Musk was an early investor who provided most of the initial financing in electric vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), assuming the position of the company's chairman. He later became the product architect, and in 2008 the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and The Boring Company, a tunnel construction company. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial intelligence company.
    #     Musk has expressed views that have made him a polarizing figure.[5] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and "endorsing an antisemitic theory",[6] the latter of which he later apologized for.[5][7] His ownership of Twitter has been similarly controversial, being marked by layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to Twitter Blue verification, but also receiving some praise for bolstering free speech.
    # """

    summary_template = """
        given the linkedin information {information},
        and their latest twitter posts {twitter_posts} I want you to create:
        1. A short summary
        2. two interesting facts about them

        Use both information from twitter and linkedin
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"], 
        template=summary_template
    )

    # Note. Using llama3
    # llm = ChatOllama(model="llama3")

    # Note. Using mistral
    # - command: ollama pull mistral
    # llm = ChatOllama(model="mistral")

    llm = ChatOpenAI(temperature=0, model="gpt-4")

    chain = summary_prompt_template | llm | StrOutputParser()

    # Note. Using mock data
    # res = chain.invoke(input={"information": mock_information})

    # Using linkedin data
    res: str = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})
    logger.success(res)
    return res


if __name__ == "__main__":
    load_dotenv()
    # ice_break_with(name="Eden Marco Udemy")
    ice_break_with(name="Harrison Chase")
    # ice_break_with(name="Soojung Shin")
    # ice_break_with("Bill Gates")
    # ice_break_with("Jinkook Choi")
    # ice_break_with("최진국")


