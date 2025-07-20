from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model=ChatOpenAI()
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive","Negative"]= Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n  {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain=prompt | model | parser2

prompt2=PromptTemplate(
    template="Write an appropriate respose to the positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt3=PromptTemplate(
    template="Write an appropriate respose to the negative feedback \n {feedback}",
    input_variables=["feedback"],
)

chain1 = prompt2 | model | parser
chain2 = prompt3 | model | parser


branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="Positive",chain1),
    (lambda x:x.sentiment=="Negative",chain2),
    RunnableLambda(lambda x: "Couldnt find the sentiment")
)

chain = classifier_chain | branch_chain


feedback_text = "I love the new features of your product!"

result = chain.invoke({"feedback": feedback_text})
print(result)

chain.get_graph().print_ascii()