from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class LLamaChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
            Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.. Utilisez trois phrases
             maximum et soyez concis dans votre réponse. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        print("DEBUG: docs", docs)
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        
        print("DEBUG: chunks", chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.9
            },
        )
        
        print("DEBUG: retriever", self.retriever)

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
    
    @staticmethod
    def format_docs(docs):
        """Convert Documents to a single string.:"""
        formatted = [
            f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
            for doc in docs
        ]
        return "\n\n" + "\n\n".join(formatted)
    
    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None