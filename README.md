# Healthcare Q&A Application in Sindhi  
**(صحت بابت سوال جواب ايپليڪيشن – سنڌي)**

---

## Overview / جائزو

**English:**  
This project is a Retrieval-Augmented Generation (RAG) based healthcare Q&A application that provides answers in the Sindhi language. It uses Streamlit for the user interface, LangChain with community vectorstore integrations and Chroma DB for document retrieval, and the Gemni API (Google Generative Language API) for generating responses. The system processes PDF documents located in the `books/` folder, applies a custom TF-IDF embedding for efficient retrieval, and supports basic question answering on topics such as Corona virus, child health, and general health.

**سنڌي:**  
ھي پروجيڪٽ Retrieval-Augmented Generation (RAG) تي ٻڌل صحت بابت سوال جواب ايپليڪيشن آھي جيڪا سنڌي ٻوليءَ ۾ جواب مهيا ڪري ٿي. اھو ايپليڪيشن Streamlit يوزر انٽرفيس، LangChain (community vectorstore انٽيگريشن) ۽ Chroma DB دستاويزن جي ريٽريول لاءِ استعمال ڪري ٿو، ۽ Gemni API (Google Generative Language API) ذريعي جواب جنريٽ ڪري ٿو. سسٽم `books/` فولڊر مان PDF دستاويز پروسيس ڪري ٿو، ڪسٽم TF-IDF ايمبيڊنگ لاڳو ڪري ٿو، ۽ ڪورونا وائرس، ٻارن جي صحت ۽ عام صحت بابت بنيادي سوالن جا جواب مهيا ڪري ٿو.

---

## Features / خصوصيتون

**English:**  
- Provides healthcare-related Q&A in Sindhi  
- Processes PDF documents to extract and index healthcare content  
- Uses a custom TF-IDF embedding for efficient document retrieval  
- Integrates with the Gemni API for generating responses  
- Offers two main sections: Q&A and About & Team

**سنڌي:**  
- سنڌي ٻوليءَ ۾ صحت بابت سوال جواب  
- PDF دستاويزن مان صحت جو مواد ڪڍي انڊيڪس ڪندو آهي  
- موثر ريٽريول لاءِ ڪسٽم TF-IDF ايمبيڊنگ استعمال ٿئي ٿي  
- جواب جنريٽ ڪرڻ لاءِ Gemni API سان انٽيگريشن  
- ٻه مکيه حصا: سوال جواب ۽ اسان بابت / ٽيم

---

## Technologies / ٽيڪنالاجيون

**English:**  
- Python 3.11  
- Streamlit  
- LangChain (with community integrations)  
- Chroma DB  
- PyPDF2  
- scikit-learn (TF-IDF)  
- Requests

**سنڌي:**  
- Python 3.11  
- Streamlit  
- LangChain (community انٽيگريشن سان)  
- Chroma DB  
- PyPDF2  
- scikit-learn (TF-IDF ايمبيڊنگ لاءِ)  
- Requests

---

## Setup Instructions / سيٽ اپ جون هدايتون

**English:**  
1. **Clone the repository:**  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment:**  
   ```bash
   python -m venv .env
   source .env/bin/activate   # For Windows: .env\Scripts\activate
   ```

3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the project:**  
   Ensure that any required configuration files are in place. The PDF documents should be placed in the `books/` folder.

**سنڌي:**  
1. **ريپوزيٽري ڪلون ڪريو:**  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **ورچوئل انوائرنمينٽ ٺاهيو ۽ فعال ڪريو:**  
   ```bash
   python -m venv .env
   source .env/bin/activate   # ونڊوز لاءِ: .env\Scripts\activate
   ```

3. **دارومدار انسٽال ڪريو:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **پروجيڪٽ ترتيب ڏيو:**  
   يقيني بڻايو تہ ضروري ترتيب واريون فائلون موجود آھن ۽ PDF دستاويز `books/` فولڊر ۾ رکيون وڃن.

---

## Running the Application / ايپليڪيشن هلائڻ

**English:**  
To launch the Streamlit application, run the following command in your terminal:
```bash
streamlit run app.py
```
This command will start the app and open it in your default browser.

**سنڌي:**  
ايپليڪيشن هلائڻ لاءِ، ٽرمينل ۾ ھيٺ ڏنل حڪم هلائو:
```bash
streamlit run app.py
```
اھا ڪمانڊ ايپليڪيشن شروع ڪندي ۽ اوھان جي براؤزر ۾ کولي ڇڏيندي.

---

## Project Structure / پروجيڪٽ جو ڍانچو

```
.
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── books/               # Folder containing PDF documents
└── .streamlit/
    └── secrets.toml     # Configuration file for API credentials
```

**English:**  
This structure includes the main app, a folder for PDFs, and the configuration directory.

**سنڌي:**  
ھي ڍانچو مکيه ايپ، PDF دستاويزن لاءِ فولڊر ۽ ترتيب واري ڊائريڪٽري تي مشتمل آھي.

---

## How It Works / ڪم ڪرڻ جو طريقو

**Document Processing / دستاويزن جي پروسيسنگ**

- **English:**  
  The application reads PDF files from the `books/` folder using PyPDF2, extracts the text, and splits it into manageable chunks using LangChain's `CharacterTextSplitter`.

- **سنڌي:**  
  ايپليڪيشن `books/` فولڊر مان PDF پڙھي ٿي، PyPDF2 ذريعي متن ڪڍي ٿي ۽ LangChain جي `CharacterTextSplitter` ذريعي ان کي ننڍن حصن ۾ ورهائي ٿي.

**Embedding and Retrieval / ايمبيڊنگ ۽ ريٽريول**

- **English:**  
  A custom TF-IDF embedding (via scikit-learn) is applied to the text, and Chroma DB is used to store and retrieve document chunks based on similarity to the query.

- **سنڌي:**  
  ڪسٽم TF-IDF ايمبيڊنگ (scikit-learn ذريعي) استعمال ڪئي ويندي آهي، ۽ Chroma DB سوال سان مشابهت جي بنياد تي دستاويزن جي حصن کي محفوظ ۽ ريٽريو ڪرڻ لاءِ استعمال ٿئي ٿي.

**Response Generation / جواب جنريشن**

- **English:**  
  The Gemni API (Google Generative Language API) is called through a custom LangChain LLM class to generate responses based on the retrieved context.

- **سنڌي:**  
  ريٽريول ٿيل مواد جي بنياد تي جواب جنريٽ ڪرڻ لاءِ Gemni API (Google Generative Language API) کي ڪسٽم LangChain LLM ڪلاس ذريعي ڪال ڪيو ويندو آهي.

---

## Usage / استعمال

**English:**  
- **Q&A Section:** Enter a healthcare-related question in Sindhi in the text box and click the "جواب حاصل ڪريو" button to get an answer.
- **About Section:** View details about the project and team members.

**سنڌي:**  
- **سوال جواب سيڪشن:** ٽيڪسٽ باڪس ۾ صحت بابت سوال سنڌيءَ ۾ داخل ڪريو ۽ "جواب حاصل ڪريو" بٽڻ تي ڪلڪ ڪريو.
- **اسان بابت سيڪشن:** پروجيڪٽ ۽ ٽيم ميمبرز بابت تفصيل ڏسو.

---

## Conclusion / نتيجو

**English:**  
This project demonstrates a multilingual healthcare Q&A system that leverages modern NLP techniques for document retrieval and response generation. It makes healthcare information accessible in Sindhi through a simple and intuitive interface.

**سنڌي:**  
ھي پروجيڪٽ جديد NLP ٽيڪنالاجيون استعمال ڪندي ٻولين واري صحت بابت سوال جواب نظام پيش ڪري ٿو، جيڪو سادي ۽ سمجهه ۾ ايندڙ انٽرفيس ذريعي سنڌي ۾ صحت جي ڄاڻ فراهم ڪري ٿو.
