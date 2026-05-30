from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering reviews questions about a pizza restaurant. 
Here are some relevant reviews: {reviews}.
Here is the question to answer: {question}.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n------------------")
    question = input("Hey there! I am Agent4 \nAsk any question about our restaurant, I am here to help you. \nType 'q' to exit this chat. \nType below: \n")
    print("\n")
    if question == 'q':
        break
    result = chain.invoke({"reviews" : [], "question" : "What is the best pizza place in town?"})
    print(result)