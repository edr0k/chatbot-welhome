# prompts.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# --- Pydantic Models for Structured Output ---

class LeadAnalysis(BaseModel):
    score: int = Field(description="Uma pontuação de 1 a 10 indicando o potencial do lead, onde 10 é muito alto.")
    intent: str = Field(description="A intenção do lead, classificada como 'Pronto para contratar', 'Buscando informações' ou 'Apenas curioso'.")
    reasoning: str = Field(description="Uma breve explicação sobre o porquê da pontuação e da intenção atribuídas.")

class LeadSummary(BaseModel):
    nome: str = Field(description="Nome do lead, se mencionado. Caso contrário, 'Não informado'.")
    quantidade_imoveis: int = Field(description="Número de imóveis que o lead possui.")
    localizacao: str = Field(description="Cidade ou região dos imóveis.")
    experiencia_previa: str = Field(description="Descrição da experiência do lead com administração de imóveis.")
    lead_score: int = Field(description="A pontuação de potencial do lead (1-10).")
    intencao: str = Field(description="A intenção classificada do lead.")
    resumo_qualitativo: str = Field(description="Um resumo de 2-3 frases sobre a principal necessidade e situação do lead.")

# --- Prompt Templates ---

prompt_chat_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente virtual da Welhome, amigável e eficiente. Seu objetivo é fazer uma qualificação inicial do lead, fazendo as perguntas necessárias uma de cada vez. Seja cordial e natural."),
    ("human", "{input}"),
])

prompt_analysis_template = ChatPromptTemplate.from_template("""
Analise o histórico de conversa com um potencial cliente da Welhome.
Com base na conversa, atribua uma pontuação de potencial e classifique a intenção do lead.
Considere fatores como número de imóveis, "dores" mencionadas e nível de experiência.
Um lead com mais imóveis e mais "dores" tem um score maior.

{format_instructions}

Histórico da Conversa:
{historico}
""")

prompt_final_summary_template = ChatPromptTemplate.from_template("""
Extraia as informações do histórico de conversa e da análise do lead para criar um resumo estruturado para o vendedor.

{format_instructions}

Histórico da Conversa:
{historico}

Análise de Score e Intenção:
- Score: {score}
- Intenção: {intencao}
""")

prompt_rag_template = ChatPromptTemplate.from_template("""
Responda a pergunta do usuário de forma natural e amigável, com base apenas no contexto fornecido.

Contexto:
{context}

Pergunta: {input}
""")