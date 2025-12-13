from langchain_openai import ChatOpenAI
from app.core.config import settings

get_llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=settings.OPENAI_API_KEY
)
