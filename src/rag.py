from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.vectorstores.utils import filter_complex_metadata
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain_community.document_transformers import (
    LongContextReorder,
)

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

from langsmith import traceable
from langsmith.run_trees import RunTree

from uuid import uuid4

from dotenv import load_dotenv
import os

from langsmith import Client

load_dotenv()
client = Client()

# Collect run ID using openai_wrapper
run_id = uuid4()
def feedback():
    client.create_feedback(
        run_id,
        key="feedback-key",
        score=1.0,
        comment="comment",
    )

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self):
        self.model = ChatOllama(model="llama2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, add_start_index=True)
        self.prompt = PromptTemplate.from_template(
            """
            You're a helpful AI assistant. You are a teaching assistant for the course of programming languages. You will only provide relevant answers to the question Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible. You also know how to speak in Romanian.
            Always say "thanks for asking!" at the end of the answer. Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
            that justify the answer. Use the following format for your final output:

            <cited_answer>
                <answer></answer>
                <citations>
                    <citation><source_id></source_id><quote></quote></citation>
                    <citation><source_id></source_id><quote></quote></citation>
                    ...
                </citations>
            </cited_answer>
            Question: {question} 
            Context: {context} 
            Answer:
            """
        )

    def ingest(self, pdf_file_path: str, threshold: float = 0.3):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
            },
        )
        prompt = hub.pull("rlm/rag-prompt")
        print("DEBUG: retriever", self.retriever)
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | prompt
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.model, self.retriever, contextualize_q_prompt
        )
        ### Answer question ###
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\
        Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
        that justify the answer. Use the following format for your final output:
        DONT FORGET TO SAY add the citation at the end of the answer.
        <cited_answer>
            <answer></answer>
            <citations>
                <citation><source_id></source_id><quote></quote></citation>
                <citation><source_id></source_id><quote></quote></citation>
                ...
            </citations>
        </cited_answer>
        {context}"""
        
        # qa_system_prompt = self.prompt
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        store = {}
        def get_session_history(session_id):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        # print("DEBUG: query", self.conversational_rag_chain.invoke(
        #                 {"input": query},
        #                 config={
        #                     "configurable": {"session_id": "abc123"}
        #                 },
        #             ))
        
        feedback()
        
        # return self.conversational_rag_chain.invoke(
        #                 {"input": query},
        #                 config={
        #                     "configurable": {"session_id": "abc123"}
        #                 },
        #             )["answer"]

        result = self.conversational_rag_chain.invoke(
                        {"input": query},
                        config={
                            "configurable": {"session_id": "abc123"}
                        },
                    )
        
        ctx = result["context"]
        def pretty_print_docs(docs):
            return f"\n{'-' * 100}\n".join(
                    [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
                )
        def pretty_print_doc(doc):
            return f'Document:\n\n ' + doc.page_content
        
        ctx_str = pretty_print_docs(ctx)

        docs = self.retriever.get_relevant_documents(query)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        
        ctx_str = pretty_print_doc(reordered_docs[0])
            
        QA_input = {
            'question': query,
            'context': ctx_str
        }

        res = nlp(QA_input)
        print(res)
        offset = 5
        
        start = res['start'] - offset
        end = res['end'] + offset
        citation = ctx_str[start:end]
        print(citation)
        final_result = f"""
        {result["answer"]}
        Citations:
        {citation}
        """
        return final_result

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None