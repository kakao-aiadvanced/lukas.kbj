import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

print("=== Search Results ===")
response = tavily.search(query="Where does Messi play right now?", max_results=3)
context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
print("Context from search:")
for item in context:
    print(f"\nURL: {item['url']}")
    print(f"Content: {item['content']}\n")

print("\n=== Search Context ===")
response_context = tavily.get_search_context(query="Where does Messi play right now?", search_depth="advanced", max_tokens=500)
print(response_context)

print("\n=== Q&A Search ===")
response_qna = tavily.qna_search(query="Where does Messi play right now?")
print(response_qna)