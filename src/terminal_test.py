from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

model = ChatOllama(model="llama2")

prompt = PromptTemplate.from_template(
    """
    <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases
     maximum et soyez concis dans votre réponse. [/INST] </s> 
    [INST] Question: {question} 
    Context: {context} 
    Answer: [/INST]
    """
)

text_splitter = CharacterTextSplitter(chunk_size=1240, chunk_overlap=1000)

loader = PyPDFLoader("docs/pdf-test.pdf")
pages = loader.load_and_split()

documents = text_splitter.split_documents(pages)
print("DEBUG: pages", len(documents))

db = Chroma.from_documents(documents, FastEmbedEmbeddings())
query = "What is the name of the article?"
docs = db.similarity_search(query)
print(docs[0].page_content)
