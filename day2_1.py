import getpass
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


#### langchain hub 에서 prompt format 가져다 쓰기
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

example_messages