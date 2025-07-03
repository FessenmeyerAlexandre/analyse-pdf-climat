import os
import uuid
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Charger la clé API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 2. Configuration du modèle
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# 3. Interface utilisateur
st.set_page_config(page_title="Analyse Climat PDF", page_icon="🌍", layout="wide")
st.title("📄 Analyse du plan de transition climatique")

# 4. Upload du fichier PDF
uploaded_file = st.file_uploader("📥 Déposez un fichier PDF", type=["pdf"])

if uploaded_file:
    filename = f"{uuid.uuid4().hex}.pdf"
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("📚 Lecture et indexation du PDF..."):
            documents = SimpleDirectoryReader(input_files=[filename]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(similarity_top_k=4)

        st.subheader("🤖 Posez vos questions (jusqu’à 40)")
        questions = []
        for i in range(40):
            q = st.text_input(f"Question {i+1}", key=f"q{i}")
            if q.strip():
                questions.append(q.strip())

        if st.button("🎯 Interroger le document") and questions:
            for i, q in enumerate(questions):
                with st.spinner(f"Traitement de la question {i+1}..."):
                    response = query_engine.query(q)
                    st.markdown(f"**Q{i+1} : {q}**")
                    st.markdown(f"📎 {response.response}")
                    st.markdown("---")

    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture ou l'analyse du PDF : {e}")
else:
    st.info("📄 Merci de déposer un fichier PDF pour commencer.")
