# rag_system.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from prompts import prompt_rag_template

def create_rag_chain(llm, embeddings):
    """Cria a chain de RAG com base em um FAQ."""
    
    # FAQ da Welhome
    faq_welhome = """
    Pergunta: Como a Welhome garante que o aluguel será pago em dia?
    Resposta: A Welhome oferece a garantia de aluguel em dia. Mesmo que o inquilino atrase, nós garantimos o repasse do valor para o proprietário na data combinada, sem custos adicionais.

    Pergunta: Quem é responsável pela manutenção do imóvel?
    Resposta: Reparos estruturais são de responsabilidade do proprietário. Pequenas manutenções do dia a dia, como troca de lâmpadas, são do inquilino. Nossa plataforma ajuda a intermediar e orçar serviços de manutenção quando necessário.

    Pergunta: Qual é o custo da taxa de administração da Welhome?
    Resposta: Nossa taxa de administração é de 8% sobre o valor do aluguel. Essa taxa cobre a gestão completa do seu imóvel, incluindo divulgação, gestão de contratos, repasse garantido e suporte.

    Pergunta: Como funciona a vistoria do imóvel?
    Resposta: Realizamos uma vistoria profissional completa, com fotos e vídeos, antes da entrada do inquilino e após a sua saída. Isso garante que o imóvel seja devolvido nas mesmas condições em que foi entregue.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([faq_welhome])
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_rag_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain