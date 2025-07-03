import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import uuid

# 1. Charger la clé API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Clé API OpenAI manquante ou vide. Merci de l’ajouter dans le fichier .env.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 2. Configuration du modèle
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# 3. Design Europlace
st.set_page_config(page_title="Analyse Climat PDF", page_icon="🌍", layout="wide")

# Couleurs et style
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #001f4d;
    }
    h1 {
        color: #002a5c;
    }
    .stButton>button {
        background-color: #002a5c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 4. Logo Europlace
logo = Image.open("fa96e2d8-c077-4c54-8469-91797574f47c.png")
st.image(logo, width=180)

st.title("📄 Analyse du plan de transition climatique")

# 5. Upload du fichier
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
        with open("historique_reponses.txt", "w", encoding="utf-8") as file:
            for i, q in enumerate(questions):
                with st.spinner(f"Traitement de la question {i+1}..."):
                    response = query_engine.query(q)
                    st.markdown(f"**Q{i+1} : {q}**")
                    st.markdown(f"📎 {response.response}")
                    st.markdown("---")
                    file.write(f"Q{i+1}: {q}\n")
                    file.write(f"Réponse: {response.response}\n")
                    file.write("-" * 50 + "\n")

else:
    st.info("📥 Merci de déposer un fichier PDF pour commencer.")
