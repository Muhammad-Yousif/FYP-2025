import os
import glob
import logging
import traceback
from typing import List, Callable

import streamlit as st
import PyPDF2
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sindhi_healthbot")


CHUNK_SIZE = 1000  # characters per chunk
TOP_K = 3  # retrieval top-k
SIM_THRESHOLD = 0.01  # min cosine similarity to consider
BOOKS_DIR_NAME = "books"  # relative to app.py

# -----------------------
# Utilities
# -----------------------
def get_base_dir() -> str:
    """
    Returns the base directory where the app file lives.
    Useful for Streamlit Cloud / deployed environments.
    """
    return os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()


# -------------------------------
# Text extraction / loading
# -------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def extract_text(path: str) -> str:
    """
    Extract text from PDF or DOCX file. Returns empty string on error.
    """
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
        else:
            logger.warning("Unsupported file type: %s", path)
    except Exception as e:
        # Log the traceback for production debugging but show friendly message to user
        logger.exception("Failed to extract text from %s: %s", path, e)
        st.error(f"Error reading {os.path.basename(path)}: {e}")
    return text


@st.cache_data(show_spinner=False, ttl=3600)
def load_documents() -> List[str]:
    """
    Load documents from the ./books folder, split into chunks and return list of text chunks.
    Raises a Streamlit stop if folder missing or no readable content.
    """
    base_dir = get_base_dir()
    books_dir = os.path.join(base_dir, BOOKS_DIR_NAME)

    # find pdf/docx files
    paths = glob.glob(os.path.join(books_dir, "*.pdf")) + glob.glob(os.path.join(books_dir, "*.docx"))

    if not paths:
        st.warning("'books/' فولڊر خالي يا دستياب ناھي. مهرباني ڪري سرور تي ڪتاب يا دستاويز رکو.")
        st.stop()

    chunks: List[str] = []
    for path in paths:
        text = extract_text(path)
        if text and text.strip():
            # chunk by characters (keeps sentence boundaries naive but effective)
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i : i + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)

    if not chunks:
        st.warning(" دستاويزن مان پڙهڻ لائق ڪو مواد نڪتو ناهي.")
        st.stop()

    logger.info("Loaded %d chunks from %d files", len(chunks), len(paths))
    return chunks


# -------------------------------
# Retriever (TF-IDF)
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever() -> Callable[[str, int], List[str]]:
    """
    Builds TF-IDF vectorizer and index vectors. Returns a retrieve(query, k) function.
    """
    texts = load_documents()
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)

    def retrieve(query: str, k: int = TOP_K) -> List[str]:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, vectors)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        results = [texts[i] for i in top_idx if sims[i] > SIM_THRESHOLD]
        logger.debug("Retrieve: q='%s' -> %d results", query, len(results))
        return results

    return retrieve


# -------------------------------
# Google Gemini wrapper
# -------------------------------
class GoogleGeminiLLM:
    """
    Simple wrapper over google-generativeai client.
    Expects secrets.toml to contain:
    [openai_gemma]
    api_key = "..."
    model = "gemini-1.5-flash"
    """

    def __init__(self):
        cfg = st.secrets.get("openai_gemma", {})
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model", "gemini-1.5-flash")

        if not self.api_key:
            st.error("Gemini API key missing in Streamlit secrets (openai_gemma.api_key).")
            st.stop()
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            logger.exception("Failed to configure google-generativeai: %s", e)
            st.error("Google Generative AI configuration failed. ڏسو لاگس.")
            st.stop()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def call(self, prompt: str, timeout: int = 30) -> str:
        """
        Calls Gemini to generate content. Retries on transient failures.
        """
        try:
            # Different genai versions may expose different methods; prefer generate_content.
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)  # keep same pattern as earlier code
            # response.text expected; otherwise try str(response)
            return getattr(response, "text", str(response))
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            raise


# -------------------------------
# QA chain: retrieval + LLM
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain():
    retrieve = build_retriever()
    llm = GoogleGeminiLLM()

    system_prompt = (
        "اوھان صحت بابت سوالن جا جواب ڏيندڙ چيٽ بوٽ آھيو.\n"
        "واپيدار اوھان کان صحت بابت سوال پڇندا آھن ۽ اوھان کي فقط books فولڊر مان ڄاڻ استعمال ڪري جواب ڏيڻا آھن.\n"
        "جواب صرف سنڌي ۾ ڏيو، احترام، سادگي ۽ وضاحت سان.\n"
        "غير اخلاقي، قانوني يا غير متعلق سوالن جا جواب نه ڏيو.\n"
    )

    def qa(inputs: dict) -> dict:
        query = inputs.get("query", "")
        docs = retrieve(query)
        context = "\n\n".join(docs) if docs else ""
        # Keep prompt compact to avoid very long input to Gemini
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nسوال: {query}\n\nجواب:"
        logger.debug("Sending prompt of length %d to Gemini", len(prompt))
        return {"result": llm.call(prompt)}

    return qa


# -------------------------------
# Streamlit UI (main)
# -------------------------------
def main():
    st.set_page_config(page_title="صحت چيٽ بوٽ", layout="centered")
    st.title("🩺 صحت بابت چيٽ بوٽ")
    st.markdown("سوال پڇو ۽ ڪتابن مان حقيقي معلومات جي بنياد تي جواب حاصل ڪريو — سنڌي ۾.")
    st.divider()

    # session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render prior messages
    for msg in st.session_state.messages:
        role_display = " واهپيدار" if msg["role"] == "user" else " چيٽ بوٽ"
        with st.chat_message(msg["role"]):
            st.markdown(f"**{role_display}:**\n{msg['content']}")

    # Suggested quick questions (two)
    st.markdown("### تجويز ڪيل سوال:")
    col1, col2 = st.columns(2)
    q1 = "روزاني جسماني مشق جا فائدن بابت ٻڌايو."
    q2 = "صحت مند غذا ۾ ڪهڙا کاڌا شامل ڪجن؟"

    clicked_question = None
    if col1.button(q1):
        clicked_question = q1
    if col2.button(q2):
        clicked_question = q2

    # Get user input: use clicked suggestion if present, else chat_input
    if clicked_question:
        user_input = clicked_question
    else:
        # streamlit chat_input does not accept value= param in newer versions
        user_input = st.chat_input("پنھنجو سوال لکو...")

    # When user provides input, process it
    if user_input and user_input.strip():
        # append and show user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"** واهپيدار:**\n{user_input}")

        # Call QA pipeline
        with st.spinner("چيٽ بوٽ جواب تيار ڪري رهيو آهي..."):
            try:
                qa = get_qa_chain()
                result = qa({"query": user_input})
                answer = result.get("result", "معاف ڪجو، مان ھن سوال جو جواب نٿو ڏئي سگهان.")
            except Exception as e:
                # Log full traceback to help debugging in production logs
                logger.error("Error during QA: %s\n%s", e, traceback.format_exc())
                answer = "معاف ڪجو، ٽيڪنيڪي مسئلو پيش آيو. مهرباني ڪري ٻيهر ڪوشش ڪريو يا بعد ۾ رابطو ڪريو."

            # append and show assistant
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(f"** چيٽ بوٽ:**\n{answer}")


    # small health-check for secrets (non-blocking)
    # if st.sidebar.button("Check Gemini config"):
    #     cfg = st.secrets.get("openai_gemma", {})
    #     if cfg.get("api_key"):
    #         st.sidebar.success("Gemini API key present (hidden).")
    #     else:
    #         st.sidebar.error("Gemini API key missing. Add it in Streamlit secrets.")

    # st.sidebar.caption("Developed by: Muhammad Faisal Jamali © 2025")


if __name__ == "__main__":
    main()

