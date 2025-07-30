import os
import glob
import streamlit as st
import PyPDF2
import docx
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.base import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
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
def load_documents() -> list[Document]:
    books_dir = os.path.join(os.path.dirname(__file__), "books")
    pdfs = glob.glob(os.path.join(books_dir, "*.pdf"))
    docxs = glob.glob(os.path.join(books_dir, "*.docx"))
    paths = pdfs + docxs

    st.write("📄 Found files:", paths)  # For debug
    st.write("📂 Current working directory:", os.getcwd())
    st.write("📂 'books' folder exists?", os.path.isdir(books_dir))

    if not paths:
        st.warning("📁 'books/' فولڊر خالي آھي. مهرباني ڪري ڪجهه PDF يا DOCX فائلون شامل ڪريو.")
        st.stop()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = []
    for path in paths:
        raw = extract_text(path)
        if raw.strip():
            chunks = splitter.split_text(raw)
            docs.extend(
                Document(page_content=chunk, metadata={"source": os.path.basename(path), "chunk": i})
                for i, chunk in enumerate(chunks)
            )

    if not docs:
        st.warning("📂 دستاويزن مان ڪوبه قابلِ پڙهڻ مواد ناھي.")
        st.stop()
    return docs

# -------------------------------
# TF-IDF Custom Embedding (Robust)
# -------------------------------
class CustomEmbeddings(Embeddings):
    def __init__(self, corpus):
        clean_corpus = [c.strip() for c in corpus if c.strip()]
        if not clean_corpus:
            raise ValueError("دستاويزن خالي آھن يا صرف غير ضروري لفظن تي مشتمل آھن.")
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(clean_corpus)

    def embed_documents(self, texts):
        try:
            return self.vectorizer.transform(texts).toarray().tolist()
        except Exception:
            return [[0.0] * len(self.vectorizer.get_feature_names_out()) for _ in texts]

    def embed_query(self, text):
        try:
            return self.vectorizer.transform([text]).toarray()[0].tolist()
        except Exception:
            return [0.0] * len(self.vectorizer.get_feature_names_out())

# -------------------------------
# Vectorstore (In-Memory)
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    try:
        docs = load_documents()
        corpus = [doc.page_content for doc in docs]
        embeddings = CustomEmbeddings(corpus)
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="books"
        )
    except ValueError as ve:
        st.error(f"⚠️ ويڪٽر ڊيٽابيس ٺاھڻ ۾ مسئلو: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"❌ ناقابلِ متوقع غلطي: {e}")
        st.stop()

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
# QA Chain with Context
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_qa_chain():
    vectorstore = get_vectorstore()
    llm = GoogleGeminiLLM()

    system_prompt = """
اوھان صحت بابت سوالن جا جواب ڏيندڙ چيٽ بوٽ آھيو
واهپيدار اوهان کان صحت بابت سوال پڇندا اوھان کي انھن سوالن جا جواب ڏيڻا آھن
سمورا جواب books نالي فولڊر مان ڏيو
صرف صحت سان لاڳاپيل سوالن جا جواب ڏيو
واهپيدار غير اخلاقي ، غير ضروري ۽ غير قانوني سوال پڇي سگھن ٿا اوھان کي انھن سوالن جا جواب ناھن ڏيڻا
اوھان کي صرف صحت سان لاڳاپيل سوالن جا جواب ڏيڻا آھن جڏھن تہ واهپيدار کي موضوع تي رھڻ جي تلقين ۽ حوصلا افزائي ڪريو
اوھان کي سڀني سوالن جا جواب سنڌي زبان ۽ رسم الخط ۾ ڏيڻا آھن
سنڌي گرامر جو خاص خيال رکو
جواب صحيح طريقي ۽ ترتيب سان ھئڻ گھرجن
اخلاقيات جو خاص خيال رکو 
دوستاڻو رويو اختيار ڪريو
نرميءَ سان جواب ڏيو
"""
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{context}\n\nسوال: {question}")
    ])

    def qa_function(inputs):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(inputs["query"])
        context = "\n".join(d.page_content for d in docs)
        final_prompt = prompt_template.format(context=context, question=inputs["query"])
        return {"result": llm.call(final_prompt)}

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
        user_input = user_input.strip()
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
