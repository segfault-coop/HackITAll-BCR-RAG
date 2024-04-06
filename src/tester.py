from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import create_citation_fuzzy_match_chain


# Get embeddings.
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="llama2:7b")

loader = PyPDFLoader("docs/docker_cheatsheet.pdf")
pages = loader.load_and_split()

# retriever = Chroma.from_documents(documents=pages, embedding=embeddings).as_retriever(
#     search_kwargs={"k": 3}
# )
texts = [
    """
    isfgigsiughsiuhgseghsdohgiusdvhsdhgsbvisdsdbuivbsdyihoushgbsevusgyvbsdbvbsdiuvbsybvisdbuvbsdnvsdbvinsdkvbsdlnvksdbvlnsdkvbkjsdbvsdvsdbvsdbvsdbvsdbhbsnvkjsdbivjsdbvsdvkjbsdvsdkbvksbvjhsbdkjvnsbvjsd vhsdvljsdbiusndlvbnvsldjbkjsd snlsdbvsdvndnvsd,sdvkbkjsbuhewbiubsdkgievlbsdibgoweuvbsoibgusbvsvbsbvubsoevbsoibvionesbvbsjkbgsjbvkjsd vkjsjbdvkjsdkj nnskjdn jn kjn sn s n n oien oien oisn gseng oesn gois ngs fopwemfiwejifwoifiuewn   uw we ub rwb rw j eron o owpgiow gwoi ion   ' ' ' ' ' ; ' ; [ ; eowgmiwgwngiwngwegdinsogoih  I am a happy little man that likes to jump off tall buildings. I enjoy it very much. wiubewubguwugwuuguigh4iwhgunweyugwen u rqui r qwuu qwurhuhr qhwurh ouqhoirhqo roib wqonr oiqwbr iqwbroqw ihqwi rjoiqh riw qboinr quriuwqr oinqorn wqoinoi q wq '// qw mq wr/ qpr [q??, !vcueqfinkdmmawknd!@#$ knroi nwui$^UO u2gr6f$^Iomu b2u3ibuiuif ew7&&&& 9()
    """
]

retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)

query = "Does the cheatsheet contain information about Docker?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

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
# print(reordered_docs)

output = chain.run(input_documents=reordered_docs, query=query)
# print("--------------------")
# print(output)
# print("--------------------")
# print(citation)

retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
)

query = "Does the cheatsheet contain information about Docker?"

docs = retriever.invoke(query)

print(docs)