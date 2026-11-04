import logging
from langchain_exa import ExaSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from app.core.config import get_settings
from app.rag.state import AgentState,RouteDecision,EvidenceGrade
from app.rag.vector_store import get_retriever
from langchain_core.prompts import ChatPromptTemplate



logger=logging.getLogger(__name__)
setting=get_settings()

__llm=None
__web_search=None

def llm():
    global __llm
    if __llm is None:
        if not setting.gemini_api_key:
            raise RuntimeError('LLM API KEY NOT CONFIGURED!')
        else:
            __llm=ChatGoogleGenerativeAI(
                model=setting.gemini_model,
                api_key=setting.gemini_api_key,
                temperature=0.2
            )
    return __llm

def web_search_tool():
    global __web_search
    
    if __web_search is None:
        if not setting.exa_api_key:
            raise RuntimeError("Web Search Api Key not Configured!")
        else:
            __web_search=ExaSearchResults(
                    num_results=5,
                    exa_api_key=setting.exa_api_key,
                    text_contents_options=False,
                    highlights=True,               
                    type="auto"
            )
    return __web_search

def add_trace(state:AgentState,message:str):
    return [*state.get("trace",[]),message]

def Route_Decision(state:AgentState):
    
    question=state['question']
    router_llm=llm().with_structured_output(RouteDecision)
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",
            "QUESTION:{question}\
            You route messages for an enterprise IT support assistant.\
                Use kb for questions about company IT policies, VPN, password reset,\
                    MFA, laptop setup, software access, security, email, devices, \
                        troubleshooting, or technology support.\
                            Use direct only for greetings, thanks, \
                                or casual chat that needs no company knowledge."),
            ('human',"{question}")
        ]
    )
    chain=prompt|router_llm
    response=chain.invoke({"question":question})
    
    state["current_querry"]=question
    state['source_used']=response.route
    state['trace']=add_trace(state,f"Router-> {response.route.upper()}")
    
    #print(f'[{state["source_used"]}]')
    
    return state

def route_after_router(state:AgentState):
    if state['source_used']=='kb':
        return "retrieve_kb"
    else:
        return "direct_answer"

def retrieve_kb(state: AgentState):
    query = state['current_querry']

    print("STEP 1: Starting KB retrieval")
    retriever = get_retriever()

    print("STEP 2: Retriever initialized")
    docs = retriever.invoke(query)

    print("STEP 3: Documents retrieved:", len(docs))

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB Retrieval -> {len(docs)} Chunks"
        )
    }
#Grade KB EVIDENCE

def grade_kb_evidence(state:AgentState):
    kb_grader_llm=llm().with_structured_output(EvidenceGrade)
    
    question=state['question']
    context="\n\n".join(
        f"Source:{doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state['kb_docs']
    )
    
    prompt=ChatPromptTemplate.from_messages([
        ("system",
        "You are the evidence grader for an IT support assistant.\
        \
        QUESTION:{question}\
            \
        PRIVATE KB EVIDENCE:{context}\
            \
        Can this private evidence answer the question?\
        if yes,then asnwer in 'good'\
        else answer weak\
        "),
        ("user","{question}")
        
    ])
    chain=prompt|kb_grader_llm
    response=chain.invoke({"question":question,"context":context})
    state['kb_grade']=response.grade
    state['trace']=add_trace(state,f'kB evidence grade-.{response.grade}')
    #print(f"kb Grade:{response.grade}")
    
    return state


def decide_after_kb_grade(state:AgentState):
    if state["kb_grade"]=='good':
        return 'generate_from_kb'
    else:
        return 'search_web'
    

def search_web(state:AgentState):
    querry=state['current_querry']
    lines=[]
    citations=[]

    response=web_search_tool().invoke(querry)

    # ExaSearchResults returns repr(exception) as a plain string on failure
    if isinstance(response,str):
        raise RuntimeError(f"Exa search failed: {response}")

    for r in response.results:
        title=r.title
        url=r.url
        # text is None when text_contents_options=False, so fall back to highlights
        content=r.text
        if isinstance(content,dict):
            content=content.get("text","")
        if not content and getattr(r,"highlights",None):
            content="\n".join(r.highlights)

        lines.append(f"Title:{title}\nURL:{url}\n\nCONTENT:{content or ''}")
        citations.append({"title":title or url,"url":url,"type":"web"})

    web_text='\n\n'.join(lines) if lines else "No web results found."
    state['web_results']=web_text
    state['citations']=citations
    state['source_used']="web"
    state['trace']=add_trace(state,'Web Fallback-> Exa Search')
    return state


    
def grade_web_evidence(state:AgentState):
    grade_web_llm=llm().with_structured_output(EvidenceGrade)
    question=state['question']
    web_results=state['web_results']
    
    
    prompt=ChatPromptTemplate.from_messages([
        ("system",
        "You are an web search evidence grader\
        \
            Question:{question}\
                \
            Web Search Evidence:{web_results}\
            \
                can this web evidence answer the question?\
                if yes, then return with 'good' \
                else return 'weak' \
                    \
                        remember to return in json!"),
        ('human',"""question:{question} web search evidence:{web_results}""")
        
    ])
    
    chain=prompt|grade_web_llm
    
    response =chain.invoke({"question":question,"web_results":web_results})
    
    state['web_grade']=response.grade
    state['trace']=add_trace(state,f"Web Evidence Grade->{response.grade.upper()}")
    return state

    
def decide_after_web_grade(state:AgentState):
    if state['web_grade']=='good':
        return 'generate_from_web'
    elif state['retry_count']<setting.max_retries:
        return 'rewrite_querry'
    else:
        return 'answer_insufficient'
    

def rewrite_querry(state:AgentState):
    question=state['question']
    retry_count=state['retry_count']+1
    
    prompt=ChatPromptTemplate.from_messages([
        ('system',
        "Rewrite the original question for better retrieval!\
            RULES:\
                1.Preserve Original Intent.\
                2.Make it specific and search friendly\
                3.Do not answer\
                4.Return only the rewritten querry.\
                    \
                Original Question:{question} "),
        ("human","{question}")
    ])
    
    chain=prompt|llm()
    response=extract_text(chain.invoke({"question":question}).content).strip()
    
    state['current_querry']=response
    state['retry_count']=retry_count
    state['trace']=add_trace(state,f"Querry Rewritten {response}")
    
    return state


def generate_from_kb(state:AgentState):
    question=state['question']
    context="\n\n".join(
        f"Source:{doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state['kb_docs']
    )
    citations=[]
    seen=set()
    
    prompt=ChatPromptTemplate.from_messages([
        ('system',
        "You are an Enterprise IT support Copilot     \
        \
        ANSWER ONLY USING PRIVATE KB TEXT! \
            \
        RULES:\
            \
            1.Beginner-Friendly Explination\
            2.Do not invent unsupported answer.\
            3.Mention that the answer is based on private KB.\
            4.Include Source TYPE :private KB\
                \
                Question:{question}    \
            PRivate KB CONTEXT: {context}"),
        ("human",'{question}')
    ])
    
    chain=prompt|llm()
    response=chain.invoke(
        {
            "question":question,
            "context":context
            })
    response=extract_text(response.content)
    
    for d in state['kb_docs']:
        src=d.metadata.get("source","Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title":src.split("/")[-1],"url":"","type":"Private Kb"})
    
    
    state['answer']=response
    state['source_used']="private_kb"
    state['citations']=citations
    state['trace']=add_trace(state,f'Answer generattion -> Private Kb')
    
    return state
    
    
def generate_from_web(state:AgentState):
    question=state['question']
    web_context=state['web_results']
    
    prompt=ChatPromptTemplate.from_messages([
        ('system',
        "You are an Enterprise IT support Copilot   \
        \
        The PRIVATE KB was insufficient, so web search was used! \
            \
            ANSWER USING ONLY THE WEB SEARCH RESULT CONTEXT    \
        RULES:\
            \
            1.Beginner-Friendly Explination\
            2.Do not invent unsupported answer.\
            3.Mention that the answer is based on EXA SEARCH.\
            4.Include Source TYPE :WEB SEARCH\
            5.If urls are present in the context, include the most useful url!    \
                \
                Question:{question}    \
            PRivate KB CONTEXT: {web_context}"),
        ("human",'{question}')
    ])
    
    chain=prompt|llm()
    response=chain.invoke(
        {
            "question":question,
            "web_context":web_context
            })
    
    
    
    state['answer']=extract_text(response.content)
    state['source_used']="web_search"
    state['trace']=add_trace(state,f'Answer generattion -> Web Search')

    
    return state

def extract_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def direct_answer(state:AgentState):
    question=state['question']
    
    prompt=ChatPromptTemplate.from_messages([
        ("system",
        "Respond Briefly and natuarlly!\
            \
                Message:{question}"),
        ("human","{question}")
    ])
    chain=prompt|llm()
    
    response=extract_text(chain.invoke({"question":question}).content)
    
    state['answer']=response
    state['source_used']="direct"
    state['trace']=add_trace(state,f'Direct Response -> No retrieval')
    
    return state

def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )
    state['answer']=answer
    state['source_used']='insufficient_evidence'
    state['trace']=add_trace(state,f'Insufficient -> Reliable Retrieval')

    return state

def build_graph():
    workflow_graph=StateGraph(AgentState)
    
    workflow_graph.add_node("Route_Decision",Route_Decision)
    workflow_graph.add_node("retrieve_kb",retrieve_kb)
    workflow_graph.add_node("grade_kb_evidence",grade_kb_evidence)
    workflow_graph.add_node("search_web",search_web)
    workflow_graph.add_node("grade_web_evidence",grade_web_evidence)
    workflow_graph.add_node("rewrite_querry",rewrite_querry)
    workflow_graph.add_node("generate_from_kb",generate_from_kb)
    workflow_graph.add_node("generate_from_web",generate_from_web)
    workflow_graph.add_node("direct_answer",direct_answer)
    workflow_graph.add_node("answer_insufficient",answer_insufficient)

    #connecting edges
    workflow_graph.add_edge(START,'Route_Decision')
    workflow_graph.add_conditional_edges('Route_Decision',
                                        route_after_router,
                                        {"retrieve_kb":"retrieve_kb",
                                        "direct_answer":"direct_answer",},)
    workflow_graph.add_edge("retrieve_kb","grade_kb_evidence")

    workflow_graph.add_conditional_edges("grade_kb_evidence",
                                        decide_after_kb_grade,
                                        {"generate_from_kb":"generate_from_kb",
                                        "search_web":"search_web",},)

    workflow_graph.add_edge("search_web","grade_web_evidence")
    workflow_graph.add_conditional_edges("grade_web_evidence",
                                        decide_after_web_grade,
                                        {'generate_from_web':'generate_from_web',
                                        "rewrite_querry":"rewrite_querry",
                                        "answer_insufficient":"answer_insufficient",},)

    workflow_graph.add_edge("rewrite_querry",'retrieve_kb')
    workflow_graph.add_edge("generate_from_web",END)
    workflow_graph.add_edge("generate_from_kb",END)
    workflow_graph.add_edge("answer_insufficient",END)
    workflow_graph.add_edge("direct_answer",END)
    return workflow_graph.compile()

agent_graph=build_graph()

#HELPER FUNCTIION

def ask(question:str):
    initial_state:AgentState={
    "question":question,
    "current_querry":question,
    "kb_docs":[],
    "web_results":"",
    "kb_grade":"",
    "web_grade":"",
    "answer":"",
    "source_used":"",
    "retry_count":0,
    "citations":[],
    "trace":[]
    }
    

    return agent_graph.invoke(initial_state)

