# main.py (Versão Interativa com Cache)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- ADICIONE ESTAS IMPORTAÇÕES PARA O CACHE ---
import langchain
from langchain.cache import SQLiteCache
# --- FIM DA ADIÇÃO ---

# Importa as funções dos nossos módulos
from chatbot import create_conversational_chain, get_session_history, analyze_lead, generate_final_summary
from rag_system import create_rag_chain

def main():
    """Função principal que orquestra a execução do case de forma interativa."""
    # Carrega as chaves de API do arquivo .env
    load_dotenv()
    
    # --- ADICIONE ESTA LINHA PARA ATIVAR O CACHE ---
    print("Ativando o cache de LLM (SQLite)...")
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    # --- FIM DA ADIÇÃO ---

    # --- INICIALIZAÇÃO DOS MODELOS ---
    print("Inicializando modelos LLM e Embeddings...")
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Modelos inicializados com sucesso.\n")

    # O restante do código continua exatamente igual...
    
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
        
        response = conversation_chain.invoke({"input": user_input}, config=config)
        print(f"Assistente: {response.content}")

    # ... (resto do código para resumo e RAG) ...

if __name__ == "__main__":
    main()