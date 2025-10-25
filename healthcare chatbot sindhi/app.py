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
# Load text from pre-provided books
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
    books_dir = os.path.join(base_dir, "books")

    paths = glob.glob(os.path.join(books_dir, "*.pdf")) + glob.glob(os.path.join(books_dir, "*.docx"))

    if not paths:
        st.warning("📁 'books/' فولڊر خالي آھي. مهرباني ڪري PDF يا DOCX فائلون شامل ڪريو.")
        st.stop()

    chunks = []
    for path in paths:
        text = extract_text(path)
        if text.strip():
            # split long text into 1000-character chunks for TF-IDF
            split_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
            chunks.extend(split_chunks)

    if not chunks:
        st.warning("📂 فائلن مان ڪوبه پڙهڻ لائق مواد ناھي.")
        st.stop()
    return chunks


# -------------------------------
# TF-IDF Retriever
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever():
    texts = load_documents()
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)

    def retrieve(query: str, k=3):
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, vectors)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        return [texts[i] for i in top_idx if sims[i] > 0.01]

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
            st.error("⚠️ Gemini API key missing in secrets.toml.")
            st.stop()

        genai.configure(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def call(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Google API error: {e}")
            raise


# -------------------------------
# QA Function (Retriever + LLM)
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
# Streamlit Chat UI
# -------------------------------
def main():
    st.set_page_config(page_title="صحت چيٽ بوٽ", layout="centered")
    st.title("🩺 صحت بابت چيٽ بوٽ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display previous chat
    for msg in st.session_state.messages:
        role = "🤖 چيٽ بوٽ" if msg["role"] == "assistant" else "🙂 واهپيدار"
        st.chat_message(msg["role"]).markdown(f"**{role}:**\n{msg['content']}")

    # Suggested questions (like ChatGPT style)
    st.markdown("### تجويز ڪيل سوال:")
    col1, col2 = st.columns(2)
    q1 = "روزاني جسماني مشق جا فائدا ڇا آھن؟"
    q2 = "صحت مند غذا ۾ ڪھڙا کاڌا شامل ڪرڻ گھرجن؟"
    if col1.button(q1):
        st.session_state.prefill = q1
    if col2.button(q2):
        st.session_state.prefill = q2

    # user text input
    user_input = st.chat_input("پنھنجو سوال لکو...", value=st.session_state.pop("prefill", ""))

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


if __name__ == "__main__":
    main()
