import os
import uvicorn
# Removido o 'httpx' - não é mais necessário
from fastapi import FastAPI, HTTPException
# ### CORREÇÃO: Importar o CORSMiddleware DE VOLTA ###
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importações do LangChain (apenas para o Groq)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================
# APP & CORS
# ============================

app = FastAPI(
    title="Contador de Histórias AI - API",
    description="Gera histórias infantis."
)

# ### CORREÇÃO: Adicionar o Middleware DE VOLTA ###
# Este bloco permite que o seu frontend (Render) fale com o seu backend (Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todos (ou mude para a URL do seu frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# API KEYS (Lidas dos Segredos do Render)
# ============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================
# MODELOS DE DADOS (Pydantic)
# ============================

class QueryInput(BaseModel):
    query: str = Field(description="O tema da história (ex: 'um dragão medroso')")

class StoryOutput(BaseModel):
    story_text: str = Field(description="O texto da história infantil gerada")

# ============================
# 1. ENDPOINT: GERAR HISTÓRIA (Texto)
# ============================

@app.post("/gerar_historia", response_model=StoryOutput)
async def gerar_historia(input_data: QueryInput):
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada.")
        
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.9 
        )
        
        system_prompt = (
            "Você é um contador de histórias infantis. Sua voz é gentil, mágica e cativante.\n"
            "Sua missão é criar uma história infantil curta (máximo 10 parágrafos) baseada no tema do usuário.\n"
            "REGRAS:\n"
            "1. A história deve ser 100% segura para crianças (sem violência, sem temas assustadores).\n"
            "2. A história deve ter uma moral ou lição positiva no final.\n"
            "3. Use linguagem simples e descritiva que uma criança possa entender.\n"
            "4. NÃO inclua títulos, apenas comece a história (ex: 'Era uma vez...')."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "O tema da história é: {query}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        print(f"🤖 Gerando história sobre: '{input_data.query}'")
        story_text = await chain.ainvoke({"query": input_data.query})
        
        return StoryOutput(story_text=story_text)
        
    except Exception as e:
        print(f"❌ Erro no Groq: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar história: {e}")

# ============================
# INICIALIZAÇÃO (Para o Render)
# ============================

@app.get("/")
def health_check():
    return {"status": "Contador de Histórias AI está no ar! 🎙️"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Iniciando Uvicorn na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
