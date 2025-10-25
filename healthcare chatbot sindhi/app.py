import os
import glob
import streamlit as st
import PyPDF2
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

# -------------------------------
# ✅ SESSION GUARD
# -------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# -------------------------------
# CONFIG
# -------------------------------
CHUNK_SIZE = 1000
TOP_K = 3
SIM_THRESHOLD = 0.01
BOOKS_DIR_NAME = "books"

# -------------------------------
# TEXT EXTRACTION
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
        st.error(f"Error reading {os.path.basename(path)}: {e}")
    return text


@st.cache_data(show_spinner=False, ttl=3600)
def load_documents():
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    books_dir = os.path.join(base_dir, BOOKS_DIR_NAME)
    paths = glob.glob(os.path.join(books_dir, "*.pdf")) + glob.glob(os.path.join(books_dir, "*.docx"))

    if not paths:
        st.warning("📁 'books/' فولڊر خالي آھي. مهرباني ڪري PDF يا DOCX فائلون شامل ڪريو.")
        st.stop()

    chunks = []
    for path in paths:
        text = extract_text(path)
        if text.strip():
            split_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            chunks.extend(split_chunks)

    if not chunks:
        st.warning("📂 فائلن مان ڪوبه پڙهڻ لائق مواد ناھي.")
        st.stop()
    return chunks


# -------------------------------
# TF-IDF RETRIEVER
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever():
    texts = load_documents()
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)

    def retrieve(query: str, k=TOP_K):
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, vectors)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        return [texts[i] for i in top_idx if sims[i] > SIM_THRESHOLD]

    return retrieve


# -------------------------------
# GOOGLE GEMINI LLM WRAPPER
# -------------------------------
class GoogleGeminiLLM:
    def __init__(self):
        cfg = st.secrets.get("openai_gemma", {})
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model", "gemini-1.5-flash")

        if not self.api_key:
            st.error("⚠️ Gemini API key missing in secrets.toml.")
            st.stop()

        genai.configure(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def call(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            else:
                return "معاف ڪجو، مان جواب حاصل ڪرڻ ۾ ناڪام رهيس."
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            return "معاف ڪجو، ٽيڪنيڪي مسئلو پيش آيو."


# -------------------------------
# QA CHAIN
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain():
    retrieve = build_retriever()
    llm = GoogleGeminiLLM()

    system_prompt = (
        "اوھان صحت بابت سوالن جا جواب ڏيندڙ چيٽ بوٽ آھيو.\n"
        "واپيدار اوھان کان صحت بابت سوال پڇندا آھن ۽ اوھان کي صرف books فولڊر مان ڄاڻ استعمال ڪري جواب ڏيڻا آھن.\n"
        "جواب صرف سنڌي ۾ ڏيو، احترام، سادگي ۽ وضاحت سان.\n"
        "غير اخلاقي، قانوني يا غير متعلق سوالن جا جواب نه ڏيو.\n"
    )

    def qa(inputs):
        docs = retrieve(inputs["query"])
        context = "\n".join(docs)
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nسوال: {inputs['query']}\n\nجواب:"
        return {"result": llm.call(prompt)}

    return qa


# -------------------------------
# STREAMLIT CHAT UI
# -------------------------------
def main():
    st.set_page_config(page_title="صحت چيٽ بوٽ", layout="centered")
    st.title("🩺 صحت بابت چيٽ بوٽ")

    # Ensure session initialized
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        role = "🤖 چيٽ بوٽ" if msg["role"] == "assistant" else "🙂 واهپيدار"
        with st.chat_message(msg["role"]):
            st.markdown(f"**{role}:**\n{msg['content']}")

    # Suggested quick questions
    st.markdown("### تجويز ڪيل سوال:")
    col1, col2 = st.columns(2)
    q1 = "روزاني جسماني مشق جا فائدا ڇا آھن؟"
    q2 = "صحت مند غذا ۾ ڪھڙا کاڌا شامل ڪرڻ گھرجن؟"

    selected_question = None
    if col1.button(q1):
        selected_question = q1
    if col2.button(q2):
        selected_question = q2

    # Add empty markdown to ensure session setup
    st.markdown("")

    user_input = selected_question or st.chat_input("پنھنجو سوال لکو...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"**🙂 واهپيدار:**\n{user_input}")

        with st.spinner("چيٽ بوٽ جواب تيار ڪري رهيو آهي..."):
            try:
                qa = get_qa_chain()
                result = qa({"query": user_input})
                answer = result.get("result", "معاف ڪجو، مان ھن سوال جو جواب نٿو ڏئي سگهان.")
            except Exception as e:
                st.error(f"❌ خامي پيش آئي: {e}")
                answer = "معاف ڪجو، ٽيڪنيڪي مسئلو پيش آيو."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(f"**🤖 چيٽ بوٽ:**\n{answer}")


if __name__ == "__main__":
    main()
