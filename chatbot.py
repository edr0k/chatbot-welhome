# chatbot.py
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser

# Importa os prompts e modelos Pydantic do nosso arquivo de prompts
from prompts import (
    prompt_chat_template,
    prompt_analysis_template,
    prompt_final_summary_template,
    LeadAnalysis,
    LeadSummary
)

# Armazenamento em memória para o histórico das conversas
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_conversational_chain(llm):
    """Cria a chain de conversação com memória."""
    runnable = prompt_chat_template | llm
    
    conversation_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return conversation_with_history

def analyze_lead(llm, conversation_history):
    """Executa a análise de score e intenção do lead."""
    output_parser = PydanticOutputParser(pydantic_object=LeadAnalysis)
    analysis_chain = prompt_analysis_template | llm | output_parser
    
    return analysis_chain.invoke({
        "historico": conversation_history,
        "format_instructions": output_parser.get_format_instructions()
    })

def generate_final_summary(llm, conversation_history, lead_analysis):
    """Gera o resumo final estruturado para o vendedor."""
    summary_parser = PydanticOutputParser(pydantic_object=LeadSummary)
    summary_chain = prompt_final_summary_template | llm | summary_parser
    
    return summary_chain.invoke({
        "historico": conversation_history,
        "score": lead_analysis.score,
        "intencao": lead_analysis.intent,
        "format_instructions": summary_parser.get_format_instructions()
    })