import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import os

# ===================== CONFIGURATION =====================
load_dotenv()
st.set_page_config(
    page_title="Assistant Administratif Marocain 🇲🇦",
    page_icon="🗂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===================== STYLES =====================
st.markdown("""
    <style>
    body {
        background-color: #ffeef7;
    }
    .main {
        background-color: #fff5fa;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0px 4px 15px rgba(255, 182, 193, 0.3);
    }
    /* Style du champ texte */
    .stTextInput > div > div > input {
        background-color: #fff;
        color: #000 !important;  /* ✅ texte noir */
        border: 2px solid #f7b6d2;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 12px;
    }
    /* Style du titre */
    h1 {
        color: #d63384;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== CHARGER LES DONNÉES =====================
with open("operations_administratives_maroc.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Création des chunks
chunks = []
for category, docs in data.items():
    for doc in docs:
        text = f"""
        ID: {doc['id']}
        Opération: {doc['operation']}
        Catégorie: {doc['categorie']}
        Documents requis: {', '.join(doc['documents_requis'])}
        Nombre de documents: {doc['nombre_documents']}
        Autorité compétente: {doc['autorite_competente']}
        Source officielle: {doc['source_officielle']}
        Frais estimés: {doc['frais_estimés']}
        Délai moyen: {doc['delai_moyen']}
        Format de sortie: {doc['format_sortie']}
        """
        chunks.append(Document(page_content=text, metadata={"categorie": doc["categorie"], "id": doc["id"]}))

# ===================== CONSTRUCTION DE LA BASE VECTORIELLE =====================
@st.cache_resource
def build_vectorstore(_docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(_docs, embeddings)

vectorstore = build_vectorstore(chunks)

# ===================== CONFIGURATION DU MODÈLE =====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ===================== INTERFACE UTILISATEUR =====================
st.title("Assistant Administratif Marocain 🇲🇦")  # ✅ drapeau placé avant le mot "Marocain"
st.markdown("Posez votre question sur les démarches administratives marocaines 💬 ")

# Champ de saisie
user_input = st.text_input("✍️ Entrez votre question :", placeholder="Ex : Comment renouveler ma CNIE ?")

if user_input:
    if user_input.lower().strip() == "exit":
        st.success("👋 Merci d’avoir utilisé l’assistant administratif. À bientôt !")
        st.stop()

    # Recherche dans la base vectorielle
    docs_similaires = vectorstore.similarity_search(user_input, k=3)

    # Contexte combiné
    context = "\n\n".join([doc.page_content for doc in docs_similaires])

    # Création du prompt final
    final_prompt = f"""
    Tu es un assistant administratif marocain expert.
    Utilise uniquement les informations suivantes pour répondre clairement à la question de l'utilisateur.

    CONTEXTE :
    {context}

    QUESTION :
    {user_input}

    Fournis une réponse concise, claire, et bien structurée.
    """

    # Génération de la réponse
    with st.spinner("💭 Génération de la réponse..."):
        response = llm.invoke(final_prompt)
        st.markdown("###  Réponse :")
        st.write(response.content)