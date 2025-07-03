import os
import uuid
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Charger la clé API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Clé API OpenAI manquante ou vide. Merci de l’ajouter dans le fichier .env.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# Configurer le modèle
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# 🖌️ Design personnalisé Europlace
st.set_page_config(
    page_title="Analyse Climat PDF",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #f7f9fc;
        color: #002147;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #002147;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.2em;
    }
    .stTextInput>div>div>input {
        background-color: white;
        border: 1px solid #c9d6e3;
        border-radius: 8px;
    }
    .stMarkdown {
        background-color: white;
        padding: 1em;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Titre avec logo
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/d/d0/Paris_Europlace_logo.svg/2560px-Paris_Europlace_logo.svg.png", width=80)
with col2:
    st.title("📄 Analyse du plan de transition climatique")
    st.write("Posez jusqu’à 40 questions sur un document PDF relatif à la stratégie climat d’une entreprise.")

# Upload PDF
uploaded_file = st.file_uploader("Upload un fichier PDF", type=["pdf"])

if uploaded_file:
    filename = f"{uuid.uuid4().hex}.pdf"
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("📚 Lecture et indexation du PDF..."):
            documents = SimpleDirectoryReader(input_files=[filename]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(similarity_top_k=4)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        st.stop()

    st.subheader("🤖 Posez vos questions (jusqu’à 40)")

    questions = []
    for i in range(40):
        question = st.text_input(f"Question {i+1}", key=f"q{i}")
        if question.strip():
            questions.append(question.strip())

    if st.button("🎯 Interroger le document") and questions:
        for i, q in enumerate(questions):
            with st.spinner(f"Traitement de la question {i+1}..."):
                try:
                    response = query_engine.query(q)
                    st.markdown(f"**Q{i+1} : {q}**")
                    st.markdown(f"📎 {response.response}")
                    st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Erreur pour la question {i+1} : {e}")

    if st.button("🔄 Réinitialiser les questions"):
        st.experimental_rerun()

else:
    st.info("📥 Merci de déposer un fichier PDF pour commencer.")
