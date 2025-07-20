from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template ="give me detailed report on given {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="summaries the given {text} in 5 bullet points",
    input_variables=["text"]        
)

model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({"topic": "3 ediots movie"})

prompt1_result = prompt1.invoke({"topic": "3 ediots movie"})
result1=model.invoke(prompt1_result)
print(result1.content)  # Output will be the detailed report on the topic

print("\n\n ************************************ \n\n")

print(result)  # Output will be the summarized report in 5 bullet points

chain.get_graph().print_ascii()  # This will show the structure of the chain