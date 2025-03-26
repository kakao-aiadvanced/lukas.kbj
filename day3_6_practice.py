from pprint import pprint
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Tavily
tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

# Build graph
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

### Index

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

### Router
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}"),
    ]
)

question_router = prompt | llm | JsonOutputParser()

### Retrieval Grader
system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Generate
system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Hallucination Grader
system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)

hallucination_grader = prompt | llm | JsonOutputParser()

### Answer Grader
system = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)

answer_grader = prompt | llm | JsonOutputParser()

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        from_websearch: whether the current documents are from web search
        retry_count: number of retries for generation
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    from_websearch: bool
    retry_count: int


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    print("\nRetrieved Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Source: {doc.metadata.get('source', 'No source')}")
        print(f"Content: {doc.page_content[:200]}...")  # Show first 200 characters
    return {"documents": documents, "question": question, "from_websearch": False}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    # Extract sources
    sources = []
    for doc in documents:
        if 'source' in doc.metadata:
            source = {
                'url': doc.metadata['source'],
                'title': doc.metadata.get('title', 'Unknown Title')
            }
            if source not in sources:
                sources.append(source)

    # RAG generation with sources
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    # Add sources to generation if available
    if sources:
        generation += "\n\nSources:\n"
        for source in sources:
            generation += f"- {source['title']}: {source['url']}\n"

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "from_websearch": state.get("from_websearch", False),
        "retry_count": retry_count + 1,
        "sources": sources
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    from_websearch = state.get("from_websearch", False)

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
            
    # If documents are from web search and still need more web search, mark as failed
    if from_websearch and web_search == "Yes":
        print("\nFAILED: not relevant")
        return {"documents": filtered_docs, "question": question, "web_search": "failed", "from_websearch": from_websearch}
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "from_websearch": from_websearch}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    print("Searching for:", state["question"])
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    print("\nWeb Search Results:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        if doc.metadata:
            print(f"Title: {doc.metadata.get('title', 'No title')}")
            print(f"Source: {doc.metadata.get('source', 'No source')}")
        print(f"Content: {doc.page_content[:200]}...")  # Show first 200 characters
    
    return {"documents": documents, "question": question, "from_websearch": True}


### Edges

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "failed":
        return "end"
    elif web_search == "Yes":
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if retry_count >= 2:
                print("\nFAILED: hallucination")
                return "failed"
            return "retry"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        if retry_count >= 2:
            print("\nFAILED: hallucination")
            return "failed"
        return "retry"

def format_result(result):
    """
    Format the workflow result for better readability

    Args:
        result (dict): The raw result from the workflow

    Returns:
        dict: Formatted result with clean structure
    """
    # Format the final result for better readability
    formatted_result = {
        'question': result['question'],
        'generation': result.get('generation', 'Failed to generate response'),
        'from_websearch': result.get('from_websearch', False),
        'retry_count': result.get('retry_count', 0),
        'web_search': result.get('web_search', 'No'),
        'documents': []
    }
    
    # Format documents
    if 'documents' in result:
        for i, doc in enumerate(result['documents'], 1):
            doc_info = {
                'title': doc.metadata.get('title', 'No title'),
                'source': doc.metadata.get('source', 'No source'),
                'content_preview': doc.page_content[:200] + '...'  # Show first 200 characters
            }
            formatted_result['documents'].append(doc_info)
    
    return formatted_result

def create_streamlit_ui(app):
    """
    Create and render the Streamlit user interface

    Args:
        app: The compiled workflow application
    """
    st.set_page_config(
        page_title="Research Assistant",
        page_icon=":orange_heart:",
    )
    st.title("Research Assistant powered by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Superfast Llama 3 inference on Groq Cloud",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            result = None
            
            # Stream the workflow execution
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
                    if key == "generate":
                        result = value

            # Format and display the result
            if result and "generation" in result:
                final_report = result["generation"]
                st.markdown(final_report)
                
                # Display sources if available
                if "sources" in result:
                    st.markdown("### Sources")
                    for source in result["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
            else:
                st.error("Failed to generate report. Please try a different topic.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()

def terminal_test(app):
    # 할루시네이션 재시도 케이스
    question = "What are the specific implementation details of GPT-4's prompt engineering mechanism?"

    result = app.invoke({"question": question, "retry_count": 0})
    print("\nFinal Result:")
    formatted_result = format_result(result)
    pprint(formatted_result, width=80, sort_dicts=False)

def main():
    # Initialize workflow
    workflow = StateGraph(GraphState)

    ### Workflow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
            "end": END
        },
    )
    workflow.add_edge("websearch", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "retry": "generate",
            "failed": END,
        },
    )

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate

    # Run the workflow
    app = workflow.compile()

    # 터미널 테스트
    # terminal_test(app)

    # Create and run Streamlit UI
    create_streamlit_ui(app)


if __name__ == "__main__":
    main()