from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import PyPDFLoader

# Get embeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = PyPDFLoader("docs/pdf-test.pdf")
pages = loader.load_and_split()

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]
retriever = Chroma.from_documents(documents=pages, embedding=embeddings).as_retriever(
    search_kwargs={"k": 3}
)

query = "Poti sa imi zici care sunt conflictele din acest document?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)
# print(docs)

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# print(reordered_docs)

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"
llm = ChatOllama(model="llama2")
stuff_prompt_override = """Given this text extracts:
-----
{context}
-----
Please answer the following question:
{query}"""

prompt = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)

# Instantiate the chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)
output = chain.run(input_documents=reordered_docs, query=query)

print(output)