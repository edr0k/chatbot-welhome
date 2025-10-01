# main.py (Versão Interativa)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import langchain
from langchain.cache import SQLiteCache
from chatbot import create_conversational_chain, get_session_history, analyze_lead, generate_final_summary
from rag_system import create_rag_chain

def main():
    """Função principal que orquestra a execução do case de forma interativa."""
    # Carrega as chaves de API do arquivo .env
    load_dotenv()

    print("Ativando o cache de LLM (SQLite)...")
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    
    # --- INICIALIZAÇÃO DOS MODELOS ---
    print("Inicializando modelos LLM e Embeddings...")
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Modelos inicializados com sucesso.\n")

    # --- PARTE 1: CHATBOT INTERATIVO ---
    conversation_chain = create_conversational_chain(llm)
    session_id = "chat_interativo_001"
    config = {"configurable": {"session_id": session_id}}

    print("--- Inicie a conversa com o assistente da Welhome ---")
    print("(Digite 'sair' ou 'fim' para terminar a conversa e gerar o resumo)")

    # Loop de conversação interativa
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "fim"]:
            print("Assistente: Obrigado pelo seu tempo! Gerando resumo da nossa conversa...")
            break
        
        # Envia a mensagem do usuário para o chatbot
        response = conversation_chain.invoke({"input": user_input}, config=config)
        print(f"Assistente: {response.content}")

    # --- ANÁLISES E RESUMO (APÓS O FIM DA CONVERSA) ---
    conversation_history = get_session_history(session_id).messages
    
    if len(conversation_history) > 2: # Garante que houve alguma conversa
        print("\n--- Análise Avançada do Lead ---")
        lead_analysis = analyze_lead(llm, conversation_history)
        print(f"Pontuação do Lead: {lead_analysis.score}/10")
        print(f"Intenção Detectada: {lead_analysis.intent}")
        print(f"Justificativa: {lead_analysis.reasoning}\n")

        print("--- Resumo Estruturado para o Vendedor ---")
        final_summary = generate_final_summary(llm, conversation_history, lead_analysis)
        print(final_summary.json(indent=2))
    else:
        print("Nenhuma conversa registrada para gerar resumo.")

    # --- PARTE 2: TESTE DO SISTEMA RAG 
    print("\n--- Você pode também testar o sistema de perguntas e respostas (RAG) ---")
    rag_chain = create_rag_chain(llm, embeddings)
    
    while True:
        user_input_rag = input("Faça uma pergunta sobre a Welhome (ou 'sair'): ")
        if user_input_rag.lower() in ["sair", "fim"]:
            break
            
        resposta_rag = rag_chain.invoke({"input": user_input_rag})
        print(f"Resposta RAG: {resposta_rag['answer']}")


if __name__ == "__main__":
    main()