from typing  import Any, TypedDict,List,Literal,Optional
from pydantic import BaseModel,Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route:Literal['kb','direct']=Field(description="Use kb for IT docs; ANswer directly for simple chats/greetings")

class EvidenceGrade(BaseModel):
    grade:Literal["good","weak"]=Field(description="good means evidence can be used to answer; weak means evidence not good enough")


class AgentState(TypedDict):
    question:str
    current_querry:str
    kb_docs:List[Document]
    web_results:str
    kb_grade:str
    web_grade:str
    answer:str
    source_used:str
    retry_count:int
    citations:List[dict]
    trace:List[Any]