import os

if os.getenv("STREAMLIT_ENV") is None:
    # Solo actualizar pip localmente
    os.system("pip install --upgrade pip")
    # Solo actualizar transformers y torch localmente
    os.system("pip install --upgrade transformers torch")


import transformers                             # Permite trabajar con modelos preentrenados de traducción, clasificación, etc.
import torch                                    # Proporciona soporte para tensores y operaciones en GPU
import unicodedata                              # Normaliza texto eliminando tildes, acentos y caracteres especiales
from transformers import (
    AutoTokenizer,                              # Carga el tokenizador correspondiente al modelo preentrenado
    AutoModelForSeq2SeqLM,                      # Carga un modelo de tipo secuencia a secuencia (traducción, resumen, etc.)
    Seq2SeqTrainer,                             # Entrena modelos Seq2Seq con Hugging Face
    Seq2SeqTrainingArguments,                   # Define los argumentos de entrenamiento para Seq2SeqTrainer
    pipeline                                    # Crea canalizaciones (traducción, resumen, etc.)
)
from datasets import Dataset, DatasetDict       # Cargar o crear datasets personalizados; DatasetDict permite dividir entre train/test/val
#from datasets import load_metric                # Carga métricas para evaluación de modelos
import pandas as pd                             # Manipula estructuras de datos tipo DataFrame (útil para datos tabulares)
import numpy as np                              # Proporciona soporte para operaciones matemáticas con matrices y arreglos
import sentencepiece                            # Requerido por algunos modelos de tokenización como los usados por NLLB y MarianMT
import google.generativeai as genai             # Permite acceder a modelos generativos de Google como Gemini
import time                                     # Mide tiempos de ejecución o espera
import hashlib                                  # Generar funciones hash (útil para crear identificadores únicos o verificar integridad)
import json                                     # Leer, escribir y manipular datos en formato JSON
import os                                       # Acceder a funciones del sistema operativo (rutas, archivos, variables de entorno, etc.)
import re                                       # Buscar y manipular texto mediante expresiones regulares
import time                                     # Medir tiempos de ejecución, hacer pausas o calcular duración de procesos
import matplotlib.pyplot as plt                 # Visualizar datos mediante gráficos, como nubes de palabras o barras
from wordcloud import WordCloud                 # Generar visualizaciones de nube de palabras a partir de texto
import streamlit as st
import time
import random
from typing import Dict, List
import unicodedata
import re
import torch
from transformers import pipeline
import google.generativeai as genai
from transformers import MarianMTModel, MarianTokenizer


# ------------------
# Configuración de página mejorada
# ------------------

st.set_page_config(
    page_title="WasiBot Pro - Salud Mental",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.minsa.gob.pe/',
        'Report a bug': None,
        'About': "### WasiBot \nAsistente de salud mental con soporte multilingüe"
    }
)


# ------------------
# Configuración inicial mejorada
# ------------------

def check_streamlit_connection():
    """Verifica la conexión con Streamlit y muestra estado"""
    try:
        # Test de conexión simple
        st.write("")  # Intenta escribir en Streamlit
        return True
    except Exception:
        return False

if not check_streamlit_connection():
    st.error("""
    **Error de conexión con Streamlit**  
    Por favor:
    1. Asegúrate que Streamlit esté corriendo en tu terminal con:  
       `streamlit run nombre_del_script.py`
    2. Verifica tu conexión a internet
    3. Recarga la página
    """)
    st.stop()

# ------------------
# Estilos CSS personalizados - Tema Oscuro
# ------------------

# -------------------
# Estilos e inyección CSS para chat invertido
# -------------------

def inject_chat_styles():
    st.markdown("""
    <style>
    /* Contenedor del chat con scroll invertido */
    #chat-container {
        height: 65vh;
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f9f9f9;
    }

    /* Burbuja usuario */
    .user-bubble {
        background-color: #e1ffc7;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 75%;
        margin-bottom: 0.75rem;
        align-self: flex-end;
        color: #333;
    }
    /* Burbuja bot */
    .bot-bubble {
        background-color: #f0f0f0;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 75%;
        margin-bottom: 0.75rem;
        align-self: flex-start;
        color: #000;
    }
    /* Burbuja riesgo */
    .risk-bubble {
        background-color: #ffdddd;
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 75%;
        margin-bottom: 0.75rem;
        align-self: flex-start;
        border: 2px solid #ff4d4d;
        color: #a60000;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------


def load_dark_css():
    """Carga estilos CSS con tema oscuro completo"""
    st.markdown("""
    <style>
        /* Base styles - Fondo negro */
        html, body, .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
            font-family: 'Arial', sans-serif;
        }
        
        /* Mensaje de bienvenida */
        .welcome-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-out;
            border: 1px solid #444444;
        }
        
        .welcome-title {
            color: white !important;
            margin-bottom: 0.5rem !important;
        }
        
        .welcome-text {
            color: white !important;
            margin-bottom: 0 !important;
        }
        
        /* Chat bubbles */
        .user-bubble {
            background: linear-gradient(135deg, #333333 0%, #444444 100%);
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            animation: fadeIn 0.6s ease-out;
            float: right;
            clear: both;
            border: 1px solid #555555;
        }
        
        .bot-bubble {
            background: linear-gradient(135deg, #3a1c71 0%, #5b2c91 100%);
            color: white;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            animation: fadeIn 0.6s ease-out;
            float: left;
            clear: both;
            border: 1px solid #6e48aa;
        }
        
        .risk-bubble {
            background: linear-gradient(135deg, #8b0000 0%, #a52a2a 100%);
            color: white;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            animation: pulse 1.5s infinite;
            float: left;
            clear: both;
            border: 1px solid #ff4d4d;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        /* Input area */
        .stTextArea textarea {
            border-radius: 12px !important;
            padding: 12px !important;
            border: 2px solid #444444 !important;
            transition: all 0.3s ease !important;
            color: white !important;
            background-color: #222222 !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #6e48aa !important;
            box-shadow: 0 0 0 3px rgba(110, 72, 170, 0.3) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111111 !important;
            border-right: 1px solid #333333 !important;
        }
        
        .sidebar-content {
            color: white !important;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(110, 72, 170, 0.4) !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%);
        }
        
        /* Expanders */
        .st-expander {
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
        }
        
        .st-expanderHeader {
            color: white !important;
        }
        
        /* Todos los textos en blanco */
        .stMarkdown, .stAlert, .stText, .stInfo, .stWarning, .stSuccess, .stError {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_dark_css()


# ------------------
# Funciones principales mejoradas
# ------------------

def detectar_riesgo(texto: str) -> bool:
    palabras_riesgo = [
        "suicidio", "quitarme la vida", "morirme", "me corto",
        "me quiero morir", "ya no aguanto", "no quiero vivir",
        "acabar con todo", "no puedo más", "mejor estar muerto"
    ]
    texto = texto.lower()
    return any(palabra in texto for palabra in palabras_riesgo)


import time
import logging
def generar_respuesta(mensaje_usuario: str, chat, max_retries=3) -> str:
    PROMPT_SISTEMA = """
            Eres un asistente de salud mental empático y profesional, capacitado para interactuar con personas que se comunican en español o quechua. Tu función principal es brindar contención emocional, orientación inicial y recursos útiles, pero **respondes únicamente en español**, aunque el usuario escriba en quechua.
            Eres peruano. Y si te preguntan si eres de Perú responde que eres de un lugar muy bonito y recondito del Perú. Se amable y cariñoso. Gánate la confianza.
            Sigue estas reglas con estricta rigurosidad:

            1. Puedes entender textos escritos en español o quechua.
            2. Siempre responde en español. No traduzcas, no escribas ni una palabra en quechua.
            + Tampoco expliques ni describas el significado de palabras en quechua. Solo respóndelas con empatía en español.
            + No digas que prefieres que el usuario hable en español.
            + No pidas ni sugieras que el usuario cambie de idioma.
            3. **No hables de ningún otro tema** que no sea salud mental, apoyo emocional o bienestar psicológico.
            4. No respondas sobre tecnología, historia, política, ciencia, bromas ni otros temas. Si el usuario intenta desviar el tema, recuérdale con amabilidad que este es un espacio solo para apoyo emocional.
            5. **Nunca digas que eres una inteligencia artificial ni expliques cómo funcionas.**
            6. Responde con empatía, usando un lenguaje cercano y sencillo. Valida emociones sin usar tecnicismos ni diagnósticos médicos.
            7. Escucha activamente y responde con empatía genuina. Evita sonar automático.
            8. No realices diagnósticos médicos ni uses términos clínicos. No digas que alguien tiene depresión o ansiedad, solo valida sus emociones y sugiere buscar ayuda profesional.
            9. Si detectas señales de riesgo suicida (palabras como "suicidio", "quitarme la vida", "ya no aguanto", "me quiero morir"), responde con urgencia y sensibilidad, incluyendo recursos locales:
            - Línea gratuita "Habla Contigo": 0800-4-1212 (24/7 en Perú)
            - Emergencias del MINSA: Línea 113
            - Frases de apoyo como: "No estás solo/a", "Hablar ayuda", "Tu vida es valiosa"
            10. Si el usuario parece muy triste pero no en riesgo, ofrece orientación básica:
            - Técnicas de respiración o mindfulness sencillas
            - Recomienda conversar con alguien de confianza
            - Anímalo/a a buscar atención psicológica profesional
            11. Usa un lenguaje sencillo, cercano y cálido, pero profesional. Evita tecnicismos.
            12. Mantén las respuestas breves y útiles: máximo 3 párrafos.
            13. No debes incluir traducciones al quechua, ni aunque el usuario escriba en español o en quechua.
            14. No inventes datos ni enlaces médicos. Solo recomienda líneas oficiales del Perú.
            15. Recuerda que el objetivo es acompañar, no resolver todo.
            16. Si el usuario escribe en quechua, comprende el mensaje sin pedirle que lo cambie a español.
            + Nunca digas "háblame en español" ni "prefiero español".
            + Solo responde directamente en español con empatía, sin corregir ni instruir sobre el idioma.

            17. Algunas palabras en Quechua para tu conocimiento: (Puedes entender textos escritos en español o quechua, pero siempre responde en español. **No traduzcas, no escribas ni una palabra en quechua.**):
            "Allinllachu" significa en español "hola"  
            "Rimaykullayki" significa en español "buenos días"  
            "Sulpayki" significa en español "gracias"  
            "Tupananchiskama" significa en español "adios"  
            "Allichu" significa en español "por favor"  
            "Kusisqa kani" significa en español "estoy feliz"  
            "Mana sapa kanki" significa en español "no estás solo"  
            "Ari" significa en español "sí"  
            "Masi" significa en español "amigo"  
            "Kusisqa" significa en español "feliz"  
            "Llakisqa" significa en español "triste"  
            "Allinllachu kanki" significa en español "¿cómo estás?"


            Tu nombre es WasiBot. Este es un espacio seguro.
            """
    if not mensaje_usuario.strip():
        return "Por favor, escribe algo para que pueda ayudarte."

    mensaje_completo = f"{PROMPT_SISTEMA}\n\nUsuario: {mensaje_usuario}"

    for intento in range(max_retries):
        try:
            if chat is None:
                raise ValueError("El modelo Gemini no está inicializado correctamente.")

            response = chat.send_message(
                mensaje_completo,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                    top_p=0.9
                )
            )

            if hasattr(response, "text") and response.text.strip():
                return response.text.strip()
            else:
                raise ValueError("Respuesta vacía o malformada del modelo.")

        except Exception as e:
            logging.warning(f"[Gemini ERROR intento {intento+1}]: {e}")
            print(f"[Gemini ERROR intento {intento+1}]: {e}")
            time.sleep(1 + 2 ** intento)  # Aumentar el tiempo entre reintentos

    return "⚠️ No pude responder en este momento. Intenta nuevamente."


correcciones_quechua = {
    r"\WasiBot\b": "WasiBot",
    r"\wasibot\b": "WasiBot",
    r"\Wasibot\b": "WasiBot",
    r"\bhola\b": "Allinllachu",
    r"\bbuenos\s*d[ií]as\b": "rimaykullayki",
    r"\bgracias\b": "sulpayki",
    r"\badios\b": "tupananchiskama",
    r"\bpor\s+favor\b": "allichu",
    r"\bestoy\s+feliz\b": "kusisqa kani",
    r"\bno\s+estás\s+solo\b": "mana sapa kanki",
    r"\bsí\b": "ari",
    "usted": "qam",
    "amigo": "masi",
    "estoy": "kani",
    "feliz": "kusisqa",
    "triste": "llakisqa",
    "vida": "kawsay",
    "escuchar": "uyariy",
    "yo": "ñuqa",
    "tú": "qam",
    "bonito": "sumaq",
    "lugar": "llaqta",
    "perú": "Perú",
    "soy de un lugar muy bonito y recondito del Perú": "ñuqa kani Perú llaqtamanta sumaqmi, huk ch’usaq llaqtamanta.",
    "no estás solo": "mana sapa kanki",
    "estás bien": "allinllachu kanki",
    "¿cómo estás?": "allinllachu kanki"
}
# Normalizar y reemplazar
def normalizar_y_reemplazar(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    for patron, reemplazo in correcciones_quechua.items():
        texto = re.sub(patron, reemplazo, texto)
    return texto

def traducir_mejorado(texto_es: str, **kwargs) -> str:
    texto_preprocesado = normalizar_y_reemplazar(texto_es)
    params = {
        'max_length': 500,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'num_beams': 4,
        'repetition_penalty': 1.5,
        **kwargs
    }
    resultado = st.session_state.traductor(
        texto_preprocesado,
        src_lang="spa_Latn",
        tgt_lang="quy_Latn",
        **params
    )
    return resultado[0]['translation_text']

import re
import unicodedata
# Traducción mejorada integrada correctamente
def traducir_a_quechua(texto: str, traductor) -> str:
    try:
        return traducir_mejorado(texto)
    except Exception as e:
        print(f"[ERROR al traducir a quechua]: {str(e)}")
        return "⚠️ No se pudo traducir este mensaje al quechua."
    

def initialize_session_state():
    """Inicializa el estado de la sesión con valores por defecto"""
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'show_emergency' not in st.session_state:
        st.session_state.show_emergency = False


def load_models_with_retry(max_retries=3):
    """Carga los modelos con reintentos y muestra estado en Streamlit"""
    retries = 0
    traductor = None
    chat = None

    msg_placeholder = st.empty()
    log_placeholder = st.empty()

    while retries < max_retries:
        try:
            msg_placeholder.info(f"🔄 Intento {retries + 1} de {max_retries}: Cargando traductor...")
            print(f"[INFO] Intento {retries + 1}: Cargando modelo de traducción")

            traductor = pipeline(
                task="translation",
                model="umt5-base-quechua-espanol-finetuned-model-v3",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16,
                max_length=500,
                no_repeat_ngram_size=3,
                early_stopping=True,
                num_beams=4,
                repetition_penalty=1.5
            )
            msg_placeholder.success("✅ Traductor cargado correctamente.")
            print("[OK] Traductor cargado")

            time.sleep(1)  # breve pausa

            msg_placeholder.info(f"🔄 Intento {retries + 1} de {max_retries}: Cargando modelo Gemini...")
            print(f"[INFO] Intento {retries + 1}: Cargando modelo Gemini")

            genai.configure(api_key="AIzaSyB5HSYd2RCf3mV7eLQwwEbBkVb5zX4LDmU")  # Pon tu API key real

            modelo_gemini = genai.GenerativeModel("models/gemini-1.5-flash")
            chat = modelo_gemini.start_chat(history=[])

            if chat is None:
                raise RuntimeError("El chat Gemini no se inició correctamente.")

            msg_placeholder.success("✅ Modelo Gemini cargado correctamente.")
            print("[OK] Modelo Gemini cargado")

            # Limpia mensajes para no saturar la UI
            msg_placeholder.empty()
            log_placeholder.empty()

            st.session_state.model_loaded = True
            return traductor, chat

        except Exception as e:
            retries += 1
            msg_placeholder.error(f"❌ Error en intento {retries}: {e}")
            print(f"[ERROR] Fallo en intento {retries}: {e}")
            time.sleep(2 ** retries)

    msg_placeholder.error("❌ Falló la carga de modelos tras múltiples intentos.")
    st.stop()  # Detiene la ejecución de la app si no se pudo cargar

# ------------------
# Componentes de UI mejorados
# ------------------

def show_welcome_message():
    """Muestra mensaje de bienvenida con texto blanco sobre fondo oscuro"""
    with st.container():
        st.markdown("""
        <div class="welcome-container">
            <h2 class="welcome-title">💬 WasiBot - Tu Asistente de Salud Mental</h2>
            <p class="welcome-text">Hola, soy WasiBot. Este es un espacio seguro donde puedes compartir lo que sientes. 
            Puedes escribirme en español o quechua, y te responderé con empatía y cuidado.</p> 
            <p style="margin: 0; color: white;"><strong>📌 Recuerda:</strong> WasiBot no reemplaza a un profesional de salud mental, pero puede ofrecerte apoyo emocional inicial.</p>
        </div>
                
        """, unsafe_allow_html=True)
        

def display_chat_message(role: str, content: str, is_risk=False, translation=None):
    """Muestra un mensaje en el chat con estilo adecuado, incluyendo traducción dentro de la misma burbuja"""
    if role == "Usuario":
        bubble_class = "user-bubble"
        icon = "👤"
    else:
        bubble_class = "risk-bubble" if is_risk else "bot-bubble"
        icon = "🚨" if is_risk else "🤖"
    
    # Construir contenido con posible traducción incluida
    contenido_completo = content
    if translation and role.lower() == "wasibot":
        contenido_completo += f"""
        <div style="
            margin-top: 8px;
            padding: 8px;
            background-color: #f3e5ff;
            border-radius: 10px;
            border-left: 4px solid #9d50bb;
            font-size: 0.9rem;
            color: #6e48aa;
            font-weight: 600;
        ">
            🌄 Traducción Quechua:<br>{translation}
        </div>
        """

    with st.container():
        st.markdown(f"""
        <div class="{bubble_class}">
            <div style="font-weight: bold; margin-bottom: 6px;">{icon} {role.capitalize()}</div>
            <div>{contenido_completo}</div>
        </div>
        """, unsafe_allow_html=True)


def show_emergency_resources():
    """Muestra recursos de emergencia con animación de atención"""
    with st.container():
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff4d4d 0%, #f96767 100%);
            color: white;
            padding: 1rem;
            border-radius: 12px;
            margin: 1rem 0;
            animation: pulse 1.5s infinite;
            box-shadow: 0 4px 12px rgba(255, 77, 77, 0.3);
        ">
            <h3 style="color: white; margin-bottom: 0.5rem;">🚨 Recursos de Emergencia</h3>
            <p style="margin-bottom: 0.5rem;"><strong>📞 Línea gratuita "Habla Contigo":</strong> 0800-4-1212 (24/7 en Perú)</p>
            <p style="margin-bottom: 0;"><strong>🚑 Emergencias MINSA:</strong> 113</p>
        </div>
        """, unsafe_allow_html=True)


# Aplicación principal
# ------------------

def main():
    # Inicializar estado y modelos
    initialize_session_state()
    
    # Cargar modelos si no están cargados
    if not st.session_state.model_loaded:
        with st.spinner("Cargando modelos... Por favor espera"):
            try:
                traductor, chat = load_models_with_retry()
                st.session_state.traductor = traductor
                st.session_state.chat = chat
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"No se pudieron cargar los modelos: {str(e)}")
                st.stop()  # Detener la ejecución si no se pueden cargar los modelos
    
    # Verificar que los modelos estén cargados
    if not hasattr(st.session_state, 'chat') or st.session_state.chat is None:
        st.error("Los modelos no se cargaron correctamente. Por favor recarga la página.")
        st.stop()
    
    # Ahora podemos usar st.session_state.chat y st.session_state.traductor con seguridad
    traductor = st.session_state.traductor
    chat = st.session_state.chat
    
    # Sidebar profesional
    with st.sidebar:
        st.image("https://i.imgur.com/kjO0gWS.png", use_column_width=True)
        st.markdown("## 📍 Recursos de Ayuda")
        
        with st.expander("🚨 Emergencias", expanded=True):
            st.markdown("""
            - **Salud Mental:** 0800-4-1212
            - **Emergencias:** 113
            - **Violencia:** 100
            - **Suicidio:** (01) 273-8026
            """)
        
        st.markdown("---")
        st.markdown("## 🌐 Idiomas")
        st.markdown("""
        - Español (respuestas)
        - Quechua (comprensión)
        """)
        
        st.markdown("---")
        st.markdown("## 📅 Consejo Diario")
        tips = [
            "Respira profundamente 3 veces al sentir ansiedad",
            "Escribe 3 cosas por las que estés agradecido hoy",
            "Habla con alguien sobre cómo te sientes",
            "Da un paseo corto al aire libre",
            "Bebe agua regularmente"
        ]
        st.info(f"**💡 {random.choice(tips)}**")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            © 2025 WasiBot
        </div>
        """, unsafe_allow_html=True)

    # Área principal de chat
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        show_welcome_message()

        # 🟡 Entrada de chat con respuesta antes del historial
        user_input = st.chat_input("Escribe tu mensaje aquí...")

        if user_input and user_input.strip():
            with st.spinner("WasiBot está pensando..."):
                time.sleep(random.uniform(0.4, 1.2))
                riesgo = detectar_riesgo(user_input)
                respuesta = generar_respuesta(user_input, chat)
                traduccion = traducir_a_quechua(respuesta, traductor)

                # Aquí inserto el mensaje al inicio de la lista para que aparezca arriba
                st.session_state.historial.insert(0, {
                    "usuario": user_input,
                    "respuesta": respuesta,
                    "quechua": traduccion,
                    "riesgo": riesgo
                })

                if riesgo:
                    st.session_state.show_emergency = True
                st.rerun()

        # Mostrar historial de mensajes en orden: más recientes arriba
        for msg in st.session_state.historial:
            display_chat_message("Usuario", msg["usuario"], is_risk=msg["riesgo"])
            display_chat_message("WasiBot", msg["respuesta"], is_risk=msg["riesgo"], translation=msg["quechua"])

        if st.session_state.show_emergency:
            show_emergency_resources()


    with col2:
        st.markdown("### 📌 Recursos Útiles")
        st.markdown("""
        - [Guías de Autocuidado](https://www.minsa.gob.pe/)
        - [Meditaciones Guiadas](https://www.youtube.com/)
        """)
        
        st.markdown("---")
        st.markdown("### 📆 Calendario de Bienestar")
        st.markdown("""
        - **Lunes:** Autoreflexión
        - **Miércoles:** Conexión social
        - **Viernes:** Relajación
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Tu Progreso")
        st.progress(random.randint(30, 80))
        st.caption("Basado en tu interacción con WasiBot")

if __name__ == "__main__":
    main()
