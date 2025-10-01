# Chatbot Assistente para Qualificação de Leads - Case Welhome

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-b504d1?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-003366?style=for-the-badge)

## 📖 Descrição do Projeto

Este projeto é a implementação de um case de Data Science para a empresa Welhome. O objetivo é desenvolver um assistente de IA para automatizar a qualificação de leads, resolvendo o problema de negócio de baixa taxa de atendimento de ligações pelos vendedores.

O chatbot interage com o lead para coletar informações, analisar seu potencial em tempo real e, ao final, gerar um resumo estruturado e acionável para a equipe de vendas, otimizando o processo de contato.

## ✨ Funcionalidades

* **Chatbot Conversacional:** Diálogo com memória para qualificação de leads, perguntando sobre quantidade de imóveis, localização e experiência prévia.
* **Análise de Potencial (Lead Scoring):** Atribui uma pontuação de 1 a 10 ao lead com base nas informações fornecidas.
* **Análise de Intenção:** Classifica a intenção do lead (ex: "Pronto para contratar", "Buscando informações").
* **Resumo Estruturado:** Gera um sumário completo em JSON com os dados do lead, score e intenção para a equipe de vendas.
* **Sistema de RAG:** Responde a perguntas frequentes (FAQ) com base em uma base de conhecimento, evitando respostas incorretas.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.11+
* **Framework de IA:** LangChain
* **LLM (Large Language Model):** Llama 3 via API da Groq
* **Embeddings:** Google Generative AI (`models/embedding-001`)
* **Vector Store:** FAISS (para o sistema RAG)
* **Validação de Dados:** Pydantic
* **Gerenciamento de Ambiente:** venv, python-dotenv

## ⚙️ Configuração do Ambiente

Siga os passos abaixo para configurar e executar o projeto localmente.

**1. Clone o Repositório**
```bash
git clone [https://github.com/seu-usuario/chatbot-welhome.git](https://github.com/seu-usuario/chatbot-welhome.git)
cd chatbot-welhome
```

**2. Crie e Ative o Ambiente Virtual**
```bash
# Crie o ambiente
python -m venv .venv

# Ative o ambiente
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux ou macOS:
source .venv/bin/activate
```

**3. Instale as Dependências**
```bash
pip install -r requirements.txt
```

**4. Configure as Chaves de API**
* Crie um arquivo chamado `.env` na raiz do projeto.
* Adicione suas chaves da Groq e do Google AI:
  ```env
  GROQ_API_KEY="sua_chave_aqui"
  GOOGLE_API_KEY="sua_chave_aqui"
  ```

## 🚀 Como Executar

Para iniciar a aplicação em modo interativo no seu terminal, execute:
```bash
python main.py
```
O programa irá inicializar e você poderá interagir com o chatbot diretamente no terminal.

## 📂 Estrutura do Projeto

```
/chatbot-welhome/
|
|-- main.py              # Ponto de entrada da aplicação, orquestra o fluxo
|-- chatbot.py           # Lógica do chatbot conversacional e análises
|-- rag_system.py        # Lógica do sistema de RAG (FAQ)
|-- prompts.py           # Central de templates de prompts e modelos Pydantic
|-- requirements.txt     # Lista de dependências do projeto
|-- .env                 # Arquivo para as chaves de API (local)
|-- .gitignore           # Arquivo para ignorar .env e .venv
```