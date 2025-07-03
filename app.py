import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Charger la clé API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ Clé API OpenAI manquante ou vide. Merci de l’ajouter dans le fichier .env.")
    st.stop()

try:
    os.environ["OPENAI_API_KEY"] = api_key
except Exception as e:
    st.error(f"❌ Erreur avec la clé API : {e}")
    st.stop()

# 2. Configuration du modèle
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# 3. Interface
st.set_page_config(page_title="Analyse Climat PDF", page_icon="🌍", layout="wide")
st.title("📄 Analyse du plan de transition climatique")

# 4. Upload de fichier
uploaded_file = st.file_uploader("Upload un fichier PDF", type=["pdf"])

if uploaded_file:
    filename = uploaded_file.name
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📚 Lecture et indexation du PDF..."):
        try:
            documents = SimpleDirectoryReader(input_files=[filename]).load_data()
        except Exception as e:
            st.error(f"Erreur lors de la lecture du PDF : {e}")
            st.stop()

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=4)

    st.subheader("🤖 Posez vos questions (jusqu’à 40)")

    questions = []
    for i in range(40):
        question = st.text_input(f"Question {i+1}", key=f"q{i}")
        if question.strip():
            questions.append(question.strip())

    if st.button("🎯 Interroger le document") and questions:
        for i, q in enumerate(questions):
            with st.spinner(f"Traitement de la question {i+1}..."):
                response = query_engine.query(q)
                st.markdown(f"**Q{i+1} : {q}**")
                st.markdown(f"📎 {response.response}")
                st.markdown("---")
else:
    st.info("📥 Merci de déposer un fichier PDF pour commencer.")
