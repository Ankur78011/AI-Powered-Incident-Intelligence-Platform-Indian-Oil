import streamlit as st
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="IOCL AI Incident Intelligence System",
    page_icon="icon.webp",
    layout="wide"
)

col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        "icon.webp",
        width=5000
    )

with col2:
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        height: 100%;
    ">
        <h1 style="
            font-size: 58px;
            font-weight: 900;
            margin-top: 20px;
            background: linear-gradient(to right, #ff9933, #ffffff, #138808);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">
        AI Incident Intelligence System
        </h1>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------
# FILE UPLOAD
# -----------------------------------
st.markdown("Upload document Or Paste Incident Report Text")

manual_text = st.text_area(
    "Paste text here",
    height=200,
    placeholder="Paste incident report, news article, audit report, investigation findings..."
)

uploaded_file = st.file_uploader(
    "Upload Incident Report (PDF/Image)",
    type=["png", "jpg", "jpeg", "pdf"]
)
analyze_button = st.button(
    "🚀 Analyze ",
    use_container_width=True
)

if analyze_button and not uploaded_file and not manual_text.strip():

    st.warning(
        "Please upload a file or enter text."
    )

    st.stop()



# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


# -----------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------

def extract_text_pdf(file):

    doc = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text

# -----------------------------------
# IMAGE OCR EXTRACTION
# -----------------------------------

def extract_text_image(file):

    image = Image.open(file)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Improve OCR quality
    image = image.convert("L")

    image = image.resize((
        image.width * 2,
        image.height * 2
    ))

    text = pytesseract.image_to_string(image)

    return text

# -----------------------------------
# LOAD HISTORICAL REPORTS
# -----------------------------------

def load_historical_reports(folder_path="data"):

    
    all_text = ""

    if not os.path.exists(folder_path):

        st.warning(
            "⚠️ 'data' folder not found. Historical reports unavailable."
        )

        return ""

    pdf_files = [
        file for file in os.listdir(folder_path)
        if file.endswith(".pdf")
    ]


    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            folder_path,
            pdf_file
        )

        try:

            doc = fitz.open(pdf_path)

   
            text = ""

            for page in doc:
                text += page.get_text()
    

            all_text += text + "\n"

        except Exception as e:

            st.error(
                f"Error reading {pdf_file}: {e}"
            )

    return all_text

# -----------------------------------
# CHUNKING
# -----------------------------------

def chunk_text(text, size=300):

    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]

# -----------------------------------
# CREATE EMBEDDINGS
# -----------------------------------

def create_embeddings(chunks):

    valid_chunks = [
        chunk for chunk in chunks
        if chunk.strip() != ""
    ]

    if len(valid_chunks) == 0:
        return None

    embeddings = np.array([
        embedding_model.encode(chunk)
        for chunk in valid_chunks
    ])

    return embeddings

# -----------------------------------
# BUILD VECTOR DATABASE
# -----------------------------------

def build_index(embeddings):

    if embeddings is None:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

# -----------------------------------
# RETRIEVE RELEVANT CHUNKS
# -----------------------------------

def retrieve(query, chunks, index, k=5):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        query_embedding,
        k
    )

    return [
        chunks[i]
        for i in indices[0]
    ]

# -----------------------------------
# INITIAL ANALYSIS
# -----------------------------------

def analyze_with_llm(text):

    prompt = f"""
You are an expert industrial incident investigator.

You are analyzing:
- operational failures
- refinery incidents
- safety issues
- industrial accidents
- process failures

Return output in this format:

SUMMARY:
Provide concise summary.

ROOT CAUSE:
Primary root cause.

CONTRIBUTING FACTORS:
- factor 1
- factor 2

PREVENTIVE ACTIONS:
- action 1
- action 2

SEVERITY:
Low / Medium / High

INCIDENT TIMELINE:
Describe sequence of events.

RISK CATEGORY:
Operational / Safety / Financial / Environmental

SIMILAR HISTORICAL PATTERNS:
Mention if similar patterns appear in historical reports.

Report Context:

every point in a new line with a gap from previous point
all the points should start from new line strictly with a line gap
SUMMARY,ROOT CAUSE,CONTRIBUTING FACTORS,PREVENTIVE ACTIONS,SEVERITY,INCIDENT TIMELINE,RISK CATEGORY,SIMILAR HISTORICAL PATTERNS
each one of them

{text}
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        if "response" in data:
            return data["response"]

        return f"❌ Error: {data}"

    except Exception as e:

        return f"❌ Exception: {str(e)}"

# -----------------------------------
# FOLLOW-UP CHAT
# -----------------------------------

def followup_chat(query, context, history):

    prompt = f"""
You are an expert industrial incident investigation assistant.

Use:
- uploaded incident report
- historical IOCL reports
- conversation history

to answer accurately.

Context:
{context}

Conversation History:
{history}

Question:
{query}

Answer professionally and clearly.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        if "response" in data:
            return data["response"]

        return f"❌ Error: {data}"

    except Exception as e:

        return f"❌ Exception: {str(e)}"

# -----------------------------------
# LOAD HISTORICAL KNOWLEDGE BASE
# -----------------------------------

historical_text = load_historical_reports()

# -----------------------------------
# MAIN LOGIC
# -----------------------------------

if uploaded_file and analyze_button or analyze_button and len(manual_text)>0 :
    # -----------------------------------
    # EXTRACT TEXT
    # -----------------------------------

    if uploaded_file:

     if uploaded_file.type == "application/pdf":
        text = extract_text_pdf(uploaded_file)

     elif uploaded_file.type.startswith("image/"):
        text = extract_text_image(uploaded_file)

     else:
        text = ""

     text = text + "\n\n" + manual_text

    else:
       text = manual_text





    # -----------------------------------
    # SHOW EXTRACTED TEXT
    # -----------------------------------

    st.subheader("📄 Extracted Text Preview")

    st.write(text[:150000])

    # -----------------------------------
    # RAG PIPELINE
    # -----------------------------------

    with st.spinner("🧠 Running AI Analysis..."):

        # Current file chunks
        current_chunks = chunk_text(text)

        # Historical chunks
        historical_chunks = chunk_text(
            historical_text
        )

        # Combine all chunks
        all_chunks = (
            current_chunks +
            historical_chunks
        )

        # Create embeddings
        embeddings = create_embeddings(
            all_chunks
        )

        if embeddings is None:

            st.error(
                "❌ No valid text extracted."
            )

            st.stop()

        # Build vector DB
        index = build_index(
            embeddings
        )

        # Retrieve relevant chunks
        relevant_chunks = retrieve(
        text[:1000],
        all_chunks,
        index
)

        # Build context
        context = " ".join(
            relevant_chunks
        )
        full_context = f"""
CURRENT DOCUMENT:
{text}

HISTORICAL CONTEXT:
{context}
"""

        # Initial analysis
        result = analyze_with_llm(
            full_context
        )

    # -----------------------------------
    # SHOW RETRIEVED CONTEXT
    # -----------------------------------



    st.subheader("Retrieved Chunks")

    for i, chunk in enumerate(relevant_chunks):
      st.write(f"Chunk {i+1}")
      st.code(chunk[:500])

    # -----------------------------------
    # SHOW ANALYSIS
    # -----------------------------------

    st.subheader("📊 Initial Incident Analysis")

    st.write(result)

    # -----------------------------------
    # CHAT MEMORY
    # -----------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # -----------------------------------
    # CHAT INTERFACE
    # -----------------------------------

    st.subheader("💬 Chat with AI Investigator")

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # User input
    user_query = st.chat_input(
        "Ask follow-up questions..."
    )

    # -----------------------------------
    # HANDLE USER QUERY
    # -----------------------------------

    if user_query:

        # Save user query
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        # Show user message
        with st.chat_message("user"):

            st.markdown(user_query)

        # AI response
        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                # Retrieve relevant chunks
                followup_chunks = retrieve(
                    user_query,
                    all_chunks,
                    index
                )

                # Build follow-up context
                followup_context = " ".join(
                    followup_chunks
                )

                # Conversation history
                history = ""

                for msg in st.session_state.messages[-8:]:

                    history += f"""
{msg['role']}:
{msg['content']}
"""

                # Generate answer
                answer = followup_chat(
                    user_query,
                    followup_context,
                    history
                )

                # Show answer
                st.markdown(answer)

                # Save answer
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

        # -----------------------------------
        # SUPPORTING CONTEXT
        # -----------------------------------

        st.subheader("📚 Supporting Retrieved Context")

        for chunk in followup_chunks:

            st.info(chunk)