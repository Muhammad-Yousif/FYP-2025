import os
import glob
import streamlit as st
import PyPDF2
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# -------------------------------
# Load Text from Files
# -------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def extract_text(path: str) -> str:
    text = ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
        elif ext in (".docx", ".doc"):
            doc = docx.Document(path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
    return text

@st.cache_data(show_spinner=False, ttl=3600)
def load_documents():
    books_dir = os.path.join(os.path.dirname(__file__), "books")
    paths = glob.glob(os.path.join(books_dir, "*.pdf")) + glob.glob(os.path.join(books_dir, "*.docx"))

    if not paths:
        st.warning("📁 'books/' فولڊر خالي آھي. مهرباني ڪري ڪجهه PDF يا DOCX فائلون شامل ڪريو.")
        st.stop()

    chunks = []
    for path in paths:
        text = extract_text(path)
        if text.strip():
            split_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            chunks.extend(split_chunks)
    if not chunks:
        st.warning("📂 دستاويزن مان ڪوبه قابلِ پڙهڻ مواد ناھي.")
        st.stop()
    return chunks

# -------------------------------
# TF-IDF Retriever (In-Memory)
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever():
    texts = load_documents()
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)

    def retrieve(query: str, k=3):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, vectors)[0]
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [texts[i] for i in top_k_idx if similarities[i] > 0.01]

    return retrieve

# -------------------------------
# Google Gemini LLM
# -------------------------------
class GoogleGeminiLLM:
    def __init__(self):
        cfg = st.secrets.get("openai_gemma", {})
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model", "gemini-1.5-flash")

        if not self.api_key:
            st.error("Missing API key for Gemini in secrets.toml.")
            st.stop()

        genai.configure(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def call(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except GoogleAPIError as e:
            st.error(f"Google API error: {e}")
            raise

# -------------------------------
# QA Chain using Retriever + LLM
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain():
    retrieve = build_retriever()
    llm = GoogleGeminiLLM()

    system_prompt = """
اوھان صحت بابت سوالن جا جواب ڏيندڙ چيٽ بوٽ آھيو
واهپيدار اوهان کان صحت بابت سوال پڇندا اوھان کي انھن سوالن جا جواب ڏيڻا آھن
سمورا جواب books نالي فولڊر مان ڏيو
صرف صحت سان لاڳاپيل سوالن جا جواب ڏيو
غير اخلاقي، غير ضروري يا قانوني سوالن جا جواب نه ڏيو
جواب سنڌي زبان ۽ رسم الخط ۾ ڏيو، احترام، سادگي ۽ وضاحت سان
"""
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{context}\n\nسوال: {question}")
    ])

    def qa_function(inputs):
        docs = retrieve(inputs["query"])
        context = "\n".join(docs)
        prompt = prompt_template.format(context=context, question=inputs["query"])
        return {"result": llm.call(prompt)}

    return qa_function

# -------------------------------
# Streamlit Chat Interface
# -------------------------------
def main():
    st.set_page_config(page_title="صحت چيٽ بوٽ", layout="centered")
    st.title("🩺 صحت بابت چيٽ بوٽ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = "🤖 چيٽ بوٽ" if msg["role"] == "assistant" else "🙂 واهپيدار"
        st.chat_message(msg["role"]).markdown(f"**{role}:**\n{msg['content']}")

    user_input = st.chat_input("پنھنجو سوال لکو...")
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(f"**🙂 واهپيدار:**\n{user_input}")

        with st.spinner("چيٽ بوٽ جواب ڏئي رهيو آهي..."):
            try:
                qa = get_qa_chain()
                result = qa({"query": user_input})
                answer = result.get("result", "معاف ڪجو، مان ھن سوال جو جواب نٿو ڏئي سگهان.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").markdown(f"**🤖 چيٽ بوٽ:**\n{answer}")
            except Exception as e:
                st.error(f"❌ خامي پيش آئي: {e}")
    elif user_input:
        st.warning("مھرباني ڪري صحيح سوال لکو.")

if __name__ == "__main__":
    main()
