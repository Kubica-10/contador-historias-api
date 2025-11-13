import os
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException
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
    description="Gera e narra histórias infantis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# API KEYS (Lidas dos Segredos do Render)
# ============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ### CORREÇÃO DO ERRO 403 ###
# Agora lemos a chave da Gemini dos segredos do Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
# Não usamos mais a URL global, vamos construí-la dentro da função

# ============================
# MODELOS DE DADOS (Pydantic)
# ============================

class QueryInput(BaseModel):
    query: str = Field(description="O tema da história (ex: 'um dragão medroso')")

class StoryOutput(BaseModel):
    story_text: str = Field(description="O texto da história infantil gerada")

class AudioInput(BaseModel):
    text_to_speak: str = Field(description="O texto que será convertido em áudio")

class AudioOutput(BaseModel):
    audio_base64: str = Field(description="O áudio (PCM) codificado em Base64")
    mime_type: str = Field(description="O tipo MIME do áudio (ex: audio/L16; rate=24000)")

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
# 2. ENDPOINT: GERAR ÁUDIO (Voz)
# ============================

@app.post("/gerar_audio", response_model=AudioOutput)
async def gerar_audio(input_data: AudioInput):
    
    # ### CORREÇÃO DO ERRO 403 ###
    # Verificamos a chave da Gemini AQUI
    if not GEMINI_API_KEY:
        print("❌ ERRO 403: GEMINI_API_KEY não configurada no Render.")
        raise HTTPException(status_code=500, detail="Chave da API de Áudio não configurada.")
        
    # Construímos a URL AQUI, usando a chave
    TTS_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_API_KEY}"
    
    print(f"🎧 Gerando áudio para: '{input_data.text_to_speak[:30]}...'")
    
    payload = {
        "contents": [{
            "parts": [{ "text": f"Diga com uma voz gentil de contador de histórias infantis: {input_data.text_to_speak}" }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Callirrhoe" } }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(TTS_API_URL, json=payload)
            
            # Se a chave for inválida, a API retorna 403 (Proibido)
            if response.status_code == 403:
                print("❌ ERRO 403: A chave da API Gemini é inválida ou não tem permissão.")
                raise HTTPException(status_code=403, detail="A chave da API de Áudio é inválida.")
                
            response.raise_for_status() # Lança erro para outros status (ex: 500)
            
            result = response.json()
            
            part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
            audio_data = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')

            if not audio_data or not mime_type:
                raise HTTPException(status_code=500, detail="API de TTS não retornou dados de áudio.")

            return AudioOutput(audio_base64=audio_data, mime_type=mime_type)

    except httpx.RequestError as e:
        print(f"❌ Erro na API de TTS (Request): {e}")
        raise HTTPException(status_code=502, detail=f"Erro de comunicação com a API de Áudio: {e}")
    except Exception as e:
        print(f"❌ Erro no processamento de TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio: {e}")

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
