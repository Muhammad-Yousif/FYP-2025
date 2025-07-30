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
from typing import Optional
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
    paths = glob.glob("books/*.pdf") + glob.glob("books/*.docx")
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
    def __init__(self, corpus: list[str]):
        clean_corpus = [c.strip() for c in corpus if c.strip()]
        if not clean_corpus:
            raise ValueError("دستاويزن خالي آھن يا صرف غير ضروري لفظن تي مشتمل آھن.")
        
        self.vectorizer = TfidfVectorizer()
        try:
            self.vectorizer.fit(clean_corpus)
        except ValueError as e:
            raise ValueError(f"TF-IDF ويڪٽرائزر ناڪام ٿيو: {e}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.vectorizer.transform(texts).toarray().tolist()
        except Exception:
            return [[0.0] * len(self.vectorizer.get_feature_names_out()) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.vectorizer.transform([text]).toarray()[0].tolist()
        except Exception:
            return [0.0] * len(self.vectorizer.get_feature_names_out())

# -------------------------------
# Vectorstore (In-Memory, Reliable)
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    try:
        docs = load_documents()
        docs = [doc for doc in docs if doc.page_content.strip()]
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
موضوع کان ٻاھر سوالن جا جواب ڏيڻ سختي سان منع آھن 
واهپيدار جديد ٽيڪنالاجي کا واقف ناھن
اوھان کي دوستاڻو رويو اختيار ڪرڻ گھرجي 
واهپيدار اڻ پڙھيل ۽ ٽيڪنيڪل اصطلاحن کان غير واقف آھن 
اوھان کي آسان ۽ عام فهم زبان ۾جواب ڏيڻ گھرجن
اگر واهپيدار غير اخلاقي رويو اختيار ڪري ٿو تہ اوھان کي اخلاق سان دوستاڻو رويو اختيار ڪرڻ گھرجي

اوھان کي سڀني سوالن جا جواب سنڌي زبان ۽ رسم الخط ۾ ڏيڻا آھن
سنڌي گرامر جو خاص خيال رکو
جواب ۾ نقطن ۽ لفظن جي غلطي کان پاسو ڪريو
جواب صحيح طريقي ۽ ترتيب سان ھئڻ گھرجن 
جواب ۾ ھر طرح جي لفظي، املاء ۽ صورتخطيءَ جي غلطي کان پاسو ڪريو
اگر سوال سنڌي زبان کان سواءِ ڪنھن ٻي زبان ۾ اچي تہ تڏھن بہ جواب سنڌي زبان ۾ ڏيو
اوھان کي ھر جواب ۾ احترام جو مظاھرو ڪرڻو آھي 
واهپيدار سان عزت ۽ احترام سان پيش اچو
اخلاقيات جو خاص خيال رکو 
دوستاڻو رويو اختيار ڪريو
نرميءَ سان جواب ڏيو
صارفين کي ڪتاب فولڊر بابت نه ٻڌايو.
ڪنھن بہ غلط سوال جو جواب عزت سان ڏيو
واهپيدارن کي پنھنجي بناوت، ٽيڪنيڪل اصطلاحن ۽ ماڊل بابت ڄاڻ نه ڏيو
اگر واهپيدار اوھانجي بناوت بابت سوال ڪري تہ ان کي صرف اھو ٻڌايو تہ مان مصنوعي ذھانت جي اصولن تي ٺھيل صحت سان لاڳاپيل سوالن جا جواب ڏيندڙ چيٽ بوٽ آھيان.
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
