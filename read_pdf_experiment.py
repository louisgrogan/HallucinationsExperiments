import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_xai import ChatXAI
from langchain_community.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

def run_experiments():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model_choice = "gpt-4o"
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model_choice)
    parser = StrOutputParser()

    # So basically we create a chain object that works the exact same way as a model object, piping the output of the model to the parser in an automatically handled way
    # All the parser does is take the output of the model and return it as a string, it's nothing special
    chain = model | parser
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    # Basically the main thing we do with prompting is we provide an initial prompt, then for RAG we take the most relevant document chunks and put it into context, and then we also put the initial prompt into the question
    prompt = ChatPromptTemplate.from_template(template)

def prep_keys():
    load_dotenv()

__template__ = """Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

class DocumentsExtractor:
    def __init__(self, resource_path,chunks=False):
        self.resource_path = resource_path
        self.file_names = os.listdir(resource_path)
        self.file_paths = [os.path.join(resource_path, file_name) for file_name in self.file_names]
        # We want to map each file path to a list of documents extracted from that file
        self.docs = [DocumentExtractor(file_path,chunks).docs for file_path in self.file_paths]
        # Then we flatten 
        self.docs = [item for sublist in self.docs for item in sublist]


class DocumentExtractor:
    def __init__(self,file_path,chunks = False):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_extension = os.path.splitext(file_path)[1]
        if self.file_extension == ".pdf":
            self.docs = self.docs_from_pdf()
        elif self.file_extension == ".txt":
            self.docs = self.docs_from_text()
        elif self.file_extension == ".csv":
            self.docs = self.docs_from_text()
        if self.docs is None:
            raise ValueError(f"Unsupported file type: {self.file}")
        # We flatten the list of lists of documents into a single list of documents
        if chunks:
            self.chunk_docs()


    def chunk_docs(self):
        temp_docs = []
        for doc in self.docs:
            print(text_splitter.split_documents(doc))
            temp_docs.extend(text_splitter.split_documents(doc))
        self.docs = temp_docs
        
    def docs_from_text(self):
        loader = loader = TextLoader(self.file_path, encoding="utf-8")
        docs = loader.load()
        return [docs]

    def docs_from_pdf(self):
        loader = PyPDFLoader(self.file_path,extract_images=True)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        return pages


class TestingLLM:
    def __init__(self,model_choice,doc_extractor):
        self.prompt = ChatPromptTemplate.from_template(__template__)
        self.doc_extractor = doc_extractor
        self.embeddings = OpenAIEmbeddings()
        self.parser = StrOutputParser()
        self.create_vector_store()
        self.setup = RunnableParallel(context=self.vector_store.as_retriever(), question=RunnablePassthrough())
        self.set_model(model_choice)


    def create_vector_store(self):
        self.vector_store  = DocArrayInMemorySearch.from_documents(
            self.doc_extractor.docs,
            embedding=self.embeddings,
        )

    def get_closest_docs(self,question):
        return vector_store.similarity_search_with_score(query="What is Project Gigabit?", k=10)[0]


    def set_model(self,model_choice):
        self.model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model = model_choice)
        self.chain = self.setup | self.prompt | self.model | self.parser