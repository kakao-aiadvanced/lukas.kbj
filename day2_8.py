import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from langchain_openai import ChatOpenAI
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
retriever_grader_prompt = PromptTemplate(
    template="""
    You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question}
    """,
    input_variables=["document", "question"]
)


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


retriever_grader = (
        retriever_grader_prompt | llm | JsonOutputParser()
                    )

gen_prompt = PromptTemplate(
    template="""
    당신은 질문 답변 작업의 보조자입니다.
    검색된 컨텍스트의 다음 부분을 사용하여 질문에 답하십시오. 답을 모른다면 모른다고 말하십시오.
    최대 3개의 문장을 사용하고 답변은 간결하게 유지하십시오
    - 질문: {question}
    - 컨텍스트: {context}
    """,
    input_variables=["context", "question"]
)


def chain_print(question: str):
    documents = retriever.invoke(question)
    doc_txt = documents[1].page_content
    result_retriever_check = retriever_grader.invoke({"question": question, "document": doc_txt})
    if result_retriever_check.get("score", "no") == 'no':
        print("연관성 없음")
        return
    chain = gen_prompt | llm | StrOutputParser()
    output = chain.invoke({"context": doc_txt, "question": question})
    print(output)


chain_print("agent moemry")
chain_print("LLM Powered Autonomous Agents")
chain_print("Tree of Thoughts")
