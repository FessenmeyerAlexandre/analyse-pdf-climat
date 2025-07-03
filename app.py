import os
import uuid
import csv
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURATION APP ---
st.set_page_config(
    page_title="Analyse Climat PDF",
    page_icon="🌍",
    layout="wide"
)

# --- FONCTION POUR ENREGISTRER CSV ---
def enregistrer_reponses_csv(donnees, nom_fichier):
    with open(nom_fichier, mode='w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["Question", "Réponse"])
        for question, reponse in donnees:
            writer.writerow([question, reponse])

# --- MOT DE PASSE ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.image("fa96e2d8-c077-4c54-8469-91797574f47c.png", width=200)
    st.title("🔐 Accès à l'application")
    mdp = st.text_input("Entrez le mot de passe", type="password")
    if st.button("Valider"):
        if mdp == "europlace2025":
            st.session_state.auth = True
            st.experimental_rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

# --- CHARGEMENT CLÉ API ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Clé API OpenAI manquante ou vide. Merci de l’ajouter dans le fichier .env.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# --- CONFIGURATION LLM ---
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# --- INTERFACE PRINCIPALE ---
st.image("fa96e2d8-c077-4c54-8469-91797574f47c.png", width=100)
st.title("📄 Analyse du plan de transition climatique")

uploaded_file = st.file_uploader("Upload un fichier PDF", type=["pdf"])

if uploaded_file:
    nom_pdf = f"{uuid.uuid4().hex}.pdf"
    with open(nom_pdf, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("📚 Lecture et indexation du PDF..."):
            documents = SimpleDirectoryReader(input_files=[nom_pdf]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(similarity_top_k=4)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        st.stop()

    st.subheader("🤖 Posez vos questions (jusqu’à 40)")
    questions = []
    for i in range(40):
        q = st.text_input(f"Question {i+1}", key=f"q{i}")
        if q.strip():
            questions.append(q.strip())

    if st.button("🎯 Interroger le document") and questions:
        reponses = []
        for i, q in enumerate(questions):
            with st.spinner(f"Traitement de la question {i+1}..."):
                try:
                    r = query_engine.query(q)
                    st.markdown(f"**Q{i+1} : {q}**")
                    st.markdown(f"📎 {r.response}")
                    st.markdown("---")
                    reponses.append((q, r.response))
                except Exception as e:
                    st.error(f"Erreur pour la question {i+1} : {e}")

        if reponses:
            if st.download_button(
                label="📥 Télécharger les réponses en CSV",
                data='\n'.join([f'{q};{r}' for q, r in reponses]),
                file_name="reponses_analyse_climat.csv",
                mime="text/csv"
            ):
                st.success("Fichier CSV prêt !")
else:
    st.info("📥 Merci de déposer un fichier PDF pour commencer.")
