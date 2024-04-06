from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import StuffDocumentsChain,LLMChain


from langchain_community.document_loaders import PyPDFLoader

from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

from langchain.retrievers.document_compressors import EmbeddingsFilter

from langchain_community.document_transformers import (
    LongContextReorder,
)

def format_docs(docs):
    return " ".join([f"{doc['content']}\n" for doc in docs])

class LLamaChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama2", temperature = 0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt_template = """
                            Given this text extracts:
                            -----
                            {context}
                            -----
                            Please answer the following question:
                            {query}
                            -----
                            """
                            
        self.prompt = PromptTemplate(
            template = self.prompt_template,
            input_variables = ["context", "query"]
        )
        self.compressor = EmbeddingsFilter(embeddings=OllamaEmbeddings(), k=10)
        self.embeddings = OllamaEmbeddings()
        
    def split_and_filter(self,input):
        docs = input["docs"]
        question = input["question"]
        split_docs = self.splitter.split_documents(docs)
        stateful_docs = self.compressor.compress_documents(split_docs, question)
        return [stateful_doc for stateful_doc in stateful_docs]
    
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)        
        # print("DEBUG: chunks", chunks)
        vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
            },
        )
        
    def ingest_context(self, query):
        docs = self.retriever.get_relevant_documents(query)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        
        print("DEBUG: reordered_docs", reordered_docs)
        
        documnet_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        
        document_variable_name = "context"
        
        llm_chain = LLMChain(llm=self.model, prompt=self.prompt)
        self.chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=documnet_prompt,
            document_variable_name=document_variable_name,
        )
        return self.chain.run(input_documents=reordered_docs, query=query)
        
    def ask(self, query: str):
        print("DEBUG: query", query)
        response = self.ingest_context(query)
        if not self.chain:
            return "Please, add a PDF document first."
        return response

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None