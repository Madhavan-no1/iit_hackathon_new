import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import logging
from transformers import GPT2TokenizerFast  # Tokenizer to count tokens
from pathlib import Path
from gtts import gTTS
import io
from googletrans import Translator
import google.generativeai as genai

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
TOKEN_LIMIT = 512  # Limit for the model
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Google GenAI Configuration
genai.configure(api_key="api---key---brother")

# Function to initialize GenAI model
def initialize_model():
    generation_config = {"temperature": 0.9}
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Function to generate content based on image and prompts
def generate_content(model, image_path, prompts):
    image_part = {"mime_type": "image/jpeg", "data": image_path.read_bytes()}
    results = []
    for prompt_text in prompts:
        prompt_parts = [prompt_text, image_part]
        response = model.generate_content(prompt_parts)
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_part = candidate.content.parts[0]
                if text_part.text:
                    results.append(f"Prompt: {prompt_text}\nDescription:\n{text_part.text}\n")
                else:
                    results.append(f"Prompt: {prompt_text}\nDescription: No valid content generated.\n")
            else:
                results.append(f"Prompt: {prompt_text}\nDescription: No content parts found.\n")
        else:
            results.append(f"Prompt: {prompt_text}\nDescription: No candidates found.\n")
    return results

# Function to translate text
def translate_text(text, lang):
    translator = Translator()
    translation = translator.translate(text, dest=lang)
    return translation.text

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Tokenizer for counting tokens (GPT-2 tokenizer can be used for approximate counting)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to check and truncate text if necessary
def truncate_text(text, limit):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > limit:
        truncated_text = tokenizer.convert_tokens_to_string(tokens[:limit])
        logging.warning(f"Input exceeded {limit} tokens. It has been truncated.")
        return truncated_text
    return text

# Function to sanitize output
def sanitize_output(text):
    clean_text = " ".join(dict.fromkeys(text.split()))  # Remove repetitive words
    return clean_text.strip()

# Set custom prompt for QA
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        logging.info("Retrieval QA chain successfully created.")
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {e}")
        return None

# Loading the model
def load_llm():
    try:
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
        logging.info("Model loaded successfully.")
        return llm
    except Exception as e:
        logging.error(f"Error loading LLM: {e}")
        return None

# QA Model Function
def qa_bot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        if llm is None:
            return None
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa
    except Exception as e:
        logging.error(f"Error in qa_bot: {e}")
        return None

# Main output function with token truncation
def final_result(query):
    qa_result = qa_bot()
    if qa_result is None:
        return {"error": "Failed to load the QA bot. Please try again."}
    
    # Truncate the query if necessary
    query = truncate_text(query, TOKEN_LIMIT)

    try:
        response = qa_result.invoke({'query': query})
        return response
    except Exception as e:
        logging.error(f"Error getting final result: {e}")
        return {"error": "Failed to retrieve an answer. Please try again later."}

# Streamlit app
st.title("Medical Chatbot with LangChain and ClariView")

# Initialize session state variables to avoid AttributeError
if 'history' not in st.session_state:
    st.session_state.history = []

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'results' not in st.session_state:
    st.session_state.results = []

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = {}

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Text-based Chat", "Upload Image", "Chat History"])

# Tab 1: Chat functionality for medical questions using LangChain
with tab1:
    st.header("Ask a Medical Question")
    
    user_question = st.text_input("Enter your medical question below:")
    
    if user_question:
        with st.spinner('Processing your question...'):
            result = final_result(user_question)
            
            if "error" in result:
                st.error(result["error"])
            else:
                answer = result['result']
                sources = result.get('source_documents', [])

                sanitized_answer = sanitize_output(answer)
                formatted_answer = f"**Answer:**\n{sanitized_answer}\n\n"

                if sources:
                    formatted_answer += "**Sources:**\n"
                    for idx, doc in enumerate(sources, 1):
                        formatted_answer += f"Source {idx}: {doc.metadata['source']}, Page: {doc.metadata['page']}\n"
                else:
                    formatted_answer += "\n**No sources found**"

                st.write(formatted_answer)

                # Append to history
                st.session_state.history.append(f"User: {user_question}")
                st.session_state.history.append(f"Bot: {sanitized_answer}")

# ClariView functionality integrated in Tab 2
with tab2:
    st.header("ClariView - Image Interpreter")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

        model = initialize_model()

        # Input for multiple prompts
        st.write("Enter prompts (one per line):")
        prompts = st.text_area("Prompts")

        if st.button("Generate Description"):
            prompts_list = [prompt.strip() for prompt in prompts.split('\n') if prompt.strip()]

            if prompts_list:
                image_path = Path("temp_image.jpg")
                results = generate_content(model, image_path, prompts_list)
                st.session_state.history.append({"image": uploaded_file, "results": results})

                Path("temp_image.jpg").unlink()

                # Display results
                st.image(st.session_state.uploaded_file, caption='Uploaded Image', use_column_width=True)
                for description in results:
                    st.write(description)
                    audio_bytes = text_to_speech(description)
                    st.audio(audio_bytes, format='audio/mp3')

                    # Translation options
                    if st.button("Translate to Tamil", key=f"translate_tamil_{description}"):
                        if description not in st.session_state.translated_text:
                            translation = translate_text(description, 'ta')
                            st.session_state.translated_text[description] = translation
                        else:
                            translation = st.session_state.translated_text[description]

                        st.write(translation)
                        tamil_audio = text_to_speech(translation, 'ta')
                        st.audio(tamil_audio, format='audio/mp3')

# Chat history section
with tab3:
    st.header("Chat History")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history):
            st.write(f"Entry {idx + 1}")
            if isinstance(entry, dict) and "image" in entry and "results" in entry:
                st.image(entry["image"], caption=f'Image {idx + 1}', use_column_width=True)
                for description in entry["results"]:
                    st.write(description)
            else:
                st.write(entry)
    else:
        st.write("No history available.")
