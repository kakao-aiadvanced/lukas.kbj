import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from langchain_openai import ChatOpenAI
from langchain import hub

from dotenv import load_dotenv

# .env 파일을 로드하여 환경 변수에 적용
load_dotenv()

### llm 셋팅
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini")

#### langchain 셋팅
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# USER_AGENT 헤더 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# 여러 웹사이트 로드
websites = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

all_docs = []
for website in websites:
    loader = WebBaseLoader(
        web_paths=(website,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
        requests_kwargs={'headers': headers}  # headers 전달
    )
    docs = loader.load()
    all_docs.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# prompt = hub.pull("rlm/rag-prompt")
prompt = PromptTemplate(
    template="""
답변 문서가 질문에 관련이 있는지 여부를 true/false로 평가 해줘.
너무 엄격하지 않게 평가해줘. 사례나 세부사항에 대해서 자세하진 않아도됨. 
평가 내용엔 평가결과와 평가 사유가 있어야해 그리고 한글로 말해, 평가 내용 포멧은 JSON으로 해줘"
          - 답변 문서 : {document}
          - 질문 : {question}
""",
    input_variables=["document", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever_grader = prompt | llm | JsonOutputParser()

question = "agent moemry"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retriever_grader.invoke({"question" : question, "document" : doc_txt}))
print(doc_txt)

question = "LLM Powered Autonomous Agents"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retriever_grader.invoke({"question" : question, "document" : doc_txt}))
print(doc_txt)

question = "Tree of Thoughts"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retriever_grader.invoke({"question" : question, "document" : doc_txt}))
print(doc_txt)
