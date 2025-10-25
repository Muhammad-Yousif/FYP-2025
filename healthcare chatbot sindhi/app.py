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
        # show file-specific error but continue
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
            # split long text into 1000-character chunks for TF-IDF indexing
            split_chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
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

    def retrieve(query: str, k: int = 3):
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, vectors)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        return [texts[i] for i in top_idx if sims[i] > 0.01]

    return retrieve


# -------------------------------
# Google Gemini LLM wrapper
# -------------------------------
class GoogleGeminiLLM:
    def __init__(self):
        cfg = st.secrets.get("openai_gemma", {})
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model", "gemini-1.5-flash")

        if not self.api_key:
            st.error("⚠️ Gemini API key missing in secrets.toml under [openai_gemma].")
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
# Streamlit Chat UI (main)
# -------------------------------
def main():
    st.set_page_config(page_title="صحت چيٽ بوٽ", layout="centered")
    st.title("🩺 صحت بابت چيٽ بوٽ")
    st.markdown(" Sindhi health Q&A — سوال پڇو ۽ ڪتابن مان جواب حاصل ڪريو.")
    st.divider()

    # session messages (preserve chat history)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render previous messages
    for msg in st.session_state.messages:
        # msg["role"] should be either "user" or "assistant"
        display_role = "🙂 واهپيدار" if msg["role"] == "user" else "🤖 چيٽ بوٽ"
        with st.chat_message(msg["role"]):
            st.markdown(f"**{display_role}:**\n{msg['content']}")

    # Suggested quick questions (two buttons)
    st.markdown("### تجويز ڪيل سوال (تڪڙو چونڊيو):")
    cols = st.columns([1, 1])
    q1 = "روزاني جسماني مشق جا فائدا ڇا آھن؟"
    q2 = "صحت مند غذا ۾ ڪھڙا کاڌا شامل ڪرڻ گھرجن؟"

    clicked_question = None
    if cols[0].button(q1):
        clicked_question = q1
    if cols[1].button(q2):
        clicked_question = q2

    # Get input: if user clicked a suggested question, use it as input; otherwise show chat_input
    if clicked_question:
        user_input = clicked_question
    else:
        user_input = st.chat_input("پنھنجو سوال لکو...")

    # Process input
    if user_input and user_input.strip():
        # append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"**🙂 واهپيدار:**\n{user_input}")

        # compute and show assistant reply
        with st.spinner("چيٽ بوٽ جواب تيار ڪري رهيو آهي..."):
            try:
                qa = get_qa_chain()
                result = qa({"query": user_input})
                answer = result.get("result", "معاف ڪجو، مان ھن سوال جو جواب نٿو ڏئي سگهان.")
            except Exception as e:
                # show safe fallback if LLM or retrieval fails
                st.error(f"❌ خامي پيش آئي: {e}")
                answer = "معاف ڪجو، ھڪ ٽيڪنيڪي مسئلو پيش آيو. مھرباني ڪري بعد ۾ ڪوشش ڪريو."

            # append and display assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 چيٽ بوٽ:**\n{answer}")

    # Sidebar information & disclaimer
    st.sidebar.title("ℹ️ معلومات")
    st.sidebar.markdown(
        """
- 📚 جواب 'books/' فولڊر مان حاصل ڪيل معلومات تي مبني آھن.
- ⚠️ **Disclaimer:** هي صرف تعليمي معلومات لاءِ آھي — طبي مسئلن لاءِ مھرباني ڪري تصديق ٿيل صحت ماهر سان رجوع ڪريو.
"""
    )
    st.sidebar.divider()
    st.sidebar.caption("Developed by: Muhammad Faisal Jamali © 2025")


if __name__ == "__main__":
    main()
