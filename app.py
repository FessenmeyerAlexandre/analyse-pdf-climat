import os
import uuid
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 🔐 1. Charger la clé API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Clé API OpenAI manquante ou vide. Merci de l’ajouter dans le fichier `.env`.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ⚙️ 2. Configurer le LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# 🎨 3. UI
st.set_page_config(page_title="Analyse Climat PDF", page_icon="🌱", layout="wide")
st.title("📄 Analyse du plan de transition climatique")
st.write("Déposez un rapport PDF pour poser jusqu’à **40 questions** sur sa stratégie climat.")

# 📂 4. Upload PDF
uploaded_file = st.file_uploader("📎 Déposez un fichier PDF ici", type=["pdf"])
if uploaded_file:
    filename = f"{uuid.uuid4().hex}.pdf"
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("📚 Lecture et indexation du document..."):
            documents = SimpleDirectoryReader(input_files=[filename]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(similarity_top_k=4)
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # ❓ Questions utilisateur
    st.subheader("🧠 Posez vos questions (max 40)")
    questions = []
    for i in range(40):
        q = st.text_input(f"Question {i+1}", key=f"q{i}")
        if q.strip():
            questions.append(q.strip())

    if st.button("🎯 Lancer l’analyse") and questions:
        st.subheader("📬 Résultats")
        responses = []
        for i, question in enumerate(questions):
            with st.spinner(f"Traitement de la question {i+1}..."):
                try:
                    response = query_engine.query(question)
                    st.markdown(f"**Q{i+1} : {question}**")
                    st.markdown(f"✅ **Réponse** : {response.response}")
                    st.markdown("---")
                    responses.append((question, response.response))
                except Exception as err:
                    st.warning(f"Erreur pour la question {i+1} : {err}")

        # 💾 Télécharger les résultats
        if responses:
            result_text = "\n\n".join([f"Q{i+1}: {q}\nRéponse: {r}" for i, (q, r) in enumerate(responses)])
            st.download_button("📥 Télécharger les réponses", result_text.encode(), file_name="résultats.txt")
else:
    st.info("⬆️ Merci de déposer un fichier PDF pour démarrer l’analyse.")
