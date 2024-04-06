from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_transformers import (
    LongContextReorder,
)

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
# QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
# res = nlp(QA_input)

def ingest(pdf_file_path: str, query:str):
    docs = PyPDFLoader(file_path = pdf_file_path).load()
    text_spillter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_spillter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings())
    retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
            },
        )
    docs = retriever.get_relevant_documents(query)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs

    
def pretty_print_docs(docs):
    return f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
def pretty_print_doc(doc):
    return f'Document:\n\n ' + doc.page_content

pdf_file = 'docs/docker_cheatsheet.pdf'
query = 'How to remove a stopped container?'
ret = ingest(pdf_file,query=query)[0:2]
# print(ret)
# better_ctx = pretty_print_docs(ret)
# print(ret[0])
better_ctx = pretty_print_docs(ret)
print(better_ctx)

# Do windowing for the context of bert
    

QA_input = {
    'question': query,
    'context': better_ctx
}
print(better_ctx)
res = nlp(QA_input)
print(res)

start = res['start']
end = res['end']
print(better_ctx[start-20:end+20])