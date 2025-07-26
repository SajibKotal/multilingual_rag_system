

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.documents import Document
import warnings
from pdf2image import convert_from_path
import pytesseract
from transformers import pipeline
from dotenv import load_dotenv
import re

load_dotenv()
warnings.filterwarnings("ignore")


# Set Tesseract path

pytesseract.pytesseract.tesseract_cmd = os.path.normpath(os.getenv("TESSERACT_EXE_PATH"))
poppler_path=os.path.normpath(os.getenv("POPPLER_PATH"))


####### Data Extract And Preprocess #####

def ocr_bangla_from_pdf(pdf_path, start=5, end=19):
    images = convert_from_path(
        pdf_path,
        dpi=400,
        first_page=start + 1,
        last_page=end,
        poppler_path=poppler_path
    )
    page_texts = []
    for img in images:
        bangla_text = pytesseract.image_to_string(img, lang='ben')
        page_texts.append(bangla_text.strip().replace("\n", " "))
    full_text = " ".join(page_texts)
    return full_text

# === Run OCR and Save Output ===
ocr_text = ocr_bangla_from_pdf("HSC26-Bangla1st-Paper.pdf", start=5, end=19)
with open("clean_text4.txt", "w", encoding="utf-8") as f:
    f.write(ocr_text)

# === Chunking ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=['।', '\n'])
chunks = splitter.create_documents([ocr_text])

# Add 'passage:' prefix as required by E5 


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.I get this information from  https://huggingface.co/intfloat/multilingual-e5-large


formatted_chunks = [
    Document(page_content=f"passage: {doc.page_content}") for doc in chunks
]


embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")   # Embeddings with E5-base


vector_store = FAISS.from_documents(formatted_chunks, embeddings)   # Build FAISS Vector Store. Saved local and reload for furthere use more.To get fast performance

# vector_store.save_local("faiss_index")       # # Save FAISS index (only once) For Long-term memory to Store PDF chunking embedding document in Vectore database


vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # For Load Embedding indexing database to get fast performance


##    Retriever    ###########

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

##################### HuggingFace LLM (M-T5) ######################################

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
# pipe = pipeline("text2text-generation", model="google/mt5-large", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)


prompt = PromptTemplate(
    template="""
    প্রশ্নটি পড়ুন এবং উত্তর দিন শুধুমাত্র নিচের প্রসঙ্গ ব্যবহার করে।
    যদি প্রসঙ্গে উত্তরটি না থাকে, তাহলে বলুন "আমি জানি না।"
    উত্তরটি সংক্ষেপে দিন এবং বারবার একই বাক্য পুনরাবৃত্তি করবেন না।

    প্রসঙ্গ:
    {context}

    প্রশ্ন: {question}
    উত্তর:
    """,
    input_variables=['context', 'question']
)

########     RAG Chain    ###########

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)



parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | StrOutputParser()

from langdetect import detect

# For De
def format_query_auto(text):
    lang = detect(text)
    print("Detected language:", "Bangla" if lang == "bn" else "English")
    return f"query: {text.strip()}"




####### maintain Short-Time memory using Chat History list to store query and answer ####

chat_history = []
print("\n  Multilingual RAG System (type 'exit' or 'quit' to stop)\n")

while True:
    user_query = input(" User Question: ") # কার কোলে গজাননের ছোটো ভাইটি
    
    if user_query.strip().lower() in ['exit', 'quit']:
        print("\n Chat ended. Goodbye!")
        break

    query = format_query_auto(user_query)
    answer = main_chain.invoke(query)

    # Clean output
    answer = re.sub(r"<extra_id_\d+>", "", answer).strip()
    if not answer.strip():
        answer = "দুঃখিত! আমার এই সম্পর্কিত জানা নেই। অন্য কি ধরনের সহযোগিতা করতে পারি.."

    # Save history
    chat_history.append({"user": user_query, "bot": answer})

    print(f"RAG Answer : {answer}\n")

# === Print Full Chat History (Optional) ===
print("\n Full Chat History:")
for turn in chat_history:
    print(f"\n User Question: {turn['user']}\n RAG Answer: {turn['bot']}")




