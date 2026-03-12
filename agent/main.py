from xml.parsers.expat import model

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

sys_msg ="""
You are expert data analyst you are given a dataset and you need to analyze it and provide insights. You should also be able to explain your findings in a clear and concise manner. In addition to the analysis you should provide the user with a summary of the key findings, KPI's and any recommendations you may have based on the data. You should also be able to answer any questions the user may have about the data and your analysis. And provide this findings in report format with tables and charts if necessary. IMPORTANT - You shouldn't include any code in your response, you should only provide the analysis and insights based on the data provided.
"""
chat_temp = ChatPromptTemplate.from_messages(
    [
        ("system", sys_msg),
        ("human", "{input}")
    ]
)



def load_model(model_name: str = "nemotron-3-nano:30b-cloud", reasoning: bool = True, model_type="chat") -> ChatOllama:
    
    if model_type == "chat":
        model = ChatOllama(model=model_name, reasoning=reasoning)

    elif model_type == "agent":
        model = create_agent(
            llm=ChatOllama(model=model_name, reasoning=reasoning),
            system_message=SystemMessage(content=sys_msg),
            tools=[]
        )
    
    return model

if __name__ == "__main__":

    import pandas as pd 

    model = load_model()

    df = pd.read_csv(r"https://raw.githubusercontent.com/Abdelrahman142/ai-data-platform/refs/heads/main/cleaned/auto_cleaned_bmw_global_sales_dataset.csv")

    response = model.invoke(chat_temp.invoke({'input': df.to_string()}))

    print(response)