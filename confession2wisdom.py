#!/usr/bin/env python3
# streamlit_app.py
"""
Streamlit App: Voice-to-Proverb Wisdom (Structured Output + TTS)
Record audio (via browser) → Transcribe → Retrieve proverbs → 
LLM returns JSON schema: { "thai_proverb": "...", "explanation": "..." } → Text-to-Speech

Usage:
  streamlit run streamlit_app.py

Models:
  Speech-to-Text: biodatlab/whisper-th-medium-combined (Thai Whisper)
  Text-to-Speech: edge-tts (Microsoft Neural Voice)
  Embeddings: BAAI/bge-m3
  LLM: Azure OpenAI
"""

import os
import json
import tempfile
import asyncio
import base64
import html

import streamlit as st
import torch
from dotenv import load_dotenv, find_dotenv

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Chinese Proverbs Wisdom",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(find_dotenv(), override=True)

# =====================================================
# Cached loaders
# =====================================================
@st.cache_resource
def load_whisper_model(model_name: str = "biodatlab/whisper-th-medium-combined"):
    """
    Load Speech-to-Text model (Whisper).
    
    Popular Thai models:
    - "biodatlab/whisper-th-medium-combined" (recommended, default)
    - "openai/whisper-small"
    - "openai/whisper-base"
    - "charsiu/thai_male" (ก้อหนึ่ง alternative)
    """
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device,
    )
    return pipe


@st.cache_resource
def load_vector_store():
    """
    Load ChromaDB vector store (ใช้ CPU เพื่อหลีกเลี่ยง double-free crash บน WSL2).
    """
    try:
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        # Fallback ถ้า deprecated
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="chinese_proverbs",
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return vector_store, retriever


@st.cache_resource
def load_llm_client():
    """
    Load OpenAI client pointing to Azure endpoint.
    """
    from openai import OpenAI

    endpoint = "https://llm-4-vision.cognitiveservices.azure.com/openai/v1/"
    api_key = os.getenv("GPT_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GPT_API_KEY in environment variables")

    client = OpenAI(base_url=endpoint, api_key=api_key)
    # model_name = "gpt-5-mini"
    model_name = "gpt-4.1"
    return client, model_name


# =====================================================
# Core functions
# =====================================================
def transcribe_audio(audio_bytes, whisper_pipe) -> str:
    """
    Transcribe audio (from st.audio_input) using Whisper.
    audio_bytes: bytes from st.audio_input()
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        result = whisper_pipe(
            tmp_path,
            generate_kwargs={"language": "<|th|>", "task": "transcribe"},
            batch_size=16,
        )
        return (result.get("text") or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def retrieve_proverbs(query: str, retriever):
    """Retrieve relevant proverbs from vector store."""
    return retriever.invoke(query)


def _format_proverbs_context(docs) -> str:
    """Format retrieved documents as context string."""
    lines = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        chinese = md.get("chinese", "")
        pinyin = md.get("pinyin", "")
        english = md.get("english", "")
        category = md.get("category", "")
        lines.append(
            f"{i}. {chinese} | {pinyin} | {english} | {category}".strip()
        )
    return "\n".join(lines)


def _extract_first_json_obj(text: str):
    # Brace-matching: หา JSON object ก้อนแรกแบบทนทานกว่า regex
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def llm_structured_proverb(user_situation: str, docs, client_and_model):
    client, model_name = client_and_model
    proverbs_context = _format_proverbs_context(docs)

    system = (
        "You are a Thai wisdom assistant. "
        """งานของคุณ:
        - เลือก 1 สุภาษิตที่เหมาะที่สุด แล้วเรียบเรียงเป็น 'สุภาษิตจีนแปลไทย' ที่คมกระชับไม่กำกวมถูกหลักไวยากรณ์
        - เขียน explanation แบบยาว 1–2 ย่อหน้า (รวม 4–8 ประโยค) โดยต้องมีครบ:
        1) สรุปแก่นของสิ่งที่ผู้ใช้พูด (สั้น ๆ 1–2 ประโยค)
        2) อธิบายชัด ๆ ว่าทำไมสุภาษิตนี้ถึงเหมาะ ไม่ต้องมีภาษาจีนหรือพินอิน
        3) เชื่อมบทเรียนของสุภาษิตกับสถานการณ์ผู้ใช้
        4) ปิดท้ายด้วย “สิ่งที่ทำได้ทันที 1 ข้อ” (ไม่ต้องลิสต์ยาว)"""
        "Return ONLY a valid JSON object with EXACTLY these keys: "
        "\"thai_proverb\" (string), \"explanation\" (string). "
        "No extra keys. No markdown. No code fences."
    )

    user = f"""
        ข้อความที่ผู้ใช้พูด:
        {user_situation}

        สุภาษิตที่ค้นเจอ (อ้างอิงเพื่อเลือกความหมาย):
        {proverbs_context}

        ตอบกลับเป็น JSON เท่านั้น ตามรูปแบบนี้:
        {{"thai_proverb":"...","explanation":"..."}}
"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=1200,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            max_completion_tokens=1200,
        )

    content = completion.choices[0].message.content or ""
    raw = content.strip()

    # 1) ลอง parse ตรง ๆ
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 2) ถ้าไม่ใช่ JSON ล้วน ให้ดึง JSON object ก้อนแรกแล้วค่อย parse
        chunk = _extract_first_json_obj(raw)
        data = json.loads(chunk) if chunk else {}

    thai_proverb = str(data.get("thai_proverb", "")).strip()
    explanation = str(data.get("explanation", "")).strip()

    # กันกรณี key หาย/ว่าง: ส่ง raw กลับมาให้เห็นเพื่อดีบักแทน “ไม่ได้รับการตอบ”
    if not thai_proverb or not explanation:
        return {
            "thai_proverb": thai_proverb or "ผลลัพธ์ไม่ครบ (ดู raw ด้านล่าง)",
            "explanation": explanation or raw or "LLM ไม่ได้ส่งข้อความกลับมา",
        }

    return {"thai_proverb": thai_proverb, "explanation": explanation}



async def text_to_speech_edge_tts(text: str) -> bytes:
    """
    Convert text to speech using Microsoft Edge TTS (Neural Voice Thai).
    Returns: MP3 bytes
    
    Voice options:
    - "th-TH-PremwadeeNeural" (female, recommended)
    - "th-TH-NiwatNeural" (female, alternative)
    """
    import edge_tts
    #  th-TH-NiwatNeural, th-TH-PremwadeeNeural 
    voice = "th-TH-NiwatNeural"
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    return audio_data


# =====================================================
# Streamlit UI
# =====================================================
def main():
    st.title("🎤 Chinese Proverbs Wisdom")
    st.caption(
        "บันทึกเสียง → แปลงเป็นข้อความ → ค้นหาสุภาษิต → "
        "แสดงสุภาษิตไทย + คำอธิบาย + อ่านออกมา 🔊"
    )
    st.divider()

    # =====================================================
    # Sidebar
    # =====================================================
    with st.sidebar:
        st.header("⚙️ ตั้งค่า")
        
        # Model selection
        st.subheader("🧠 เลือก Speech-to-Text Model")
        model_options = {
            "Thai Whisper large (recommended)": "biodatlab/whisper-th-large-v3-combined",
            "OpenAI Whisper Small": "openai/whisper-small",
            "OpenAI Whisper Base": "openai/whisper-base",
        }
        selected_model = st.selectbox(
            "Model",
            options=model_options.keys(),
            index=0,
            help="เปลี่ยน Speech-to-Text model ได้ที่นี่"
        )
        selected_model_name = model_options[selected_model]

        if "models_loaded" not in st.session_state:
            st.session_state.models_loaded = False

        if st.button("🔄 โหลดโมเดล", type="primary", use_container_width=True):
            try:
                st.session_state.whisper_pipe = load_whisper_model(selected_model_name)
                st.session_state.vector_store, st.session_state.retriever = load_vector_store()
                st.session_state.client_and_model = load_llm_client()
                st.session_state.models_loaded = True
                st.session_state.selected_model = selected_model
                st.success(f"✅ โหลด {selected_model} สำเร็จ")
            except Exception as e:
                st.session_state.models_loaded = False
                st.error(f"❌ โหลดโมเดลไม่สำเร็จ:\n{e}")

        if st.session_state.models_loaded:
            st.success(f"✅ พร้อมใช้งาน ({st.session_state.selected_model})")
        else:
            st.warning("⚠️ กด 'โหลดโมเดล' ก่อน")

    # =====================================================
    # Main content (2 columns)
    # =====================================================
    col1, col2 = st.columns([1, 1])

    # ===== Col 1: Audio Input =====
    with col1:
        st.subheader("🎙️ บันทึกเสียง")
        st.caption("ใช้ไมค์ของ browser (Windows host บน WSL2)")

        audio = st.audio_input(
            "กดเพื่ออัดเสียง",
            disabled=not st.session_state.models_loaded,
        )

        if audio and st.button(
            "⚡ ถอดเสียงเป็นข้อความ",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("กำลังถอดเสียง..."):
                st.session_state.transcribed_text = transcribe_audio(
                    audio.getvalue(), st.session_state.whisper_pipe
                )
                st.success("✅ ถอดเสียงเสร็จแล้ว")

    # ===== Col 2: Text Input & Processing =====
    with col2:
        st.subheader("📝 ข้อความที่ถอดได้")

        if "transcribed_text" in st.session_state:
            edited_text = st.text_area(
                "แก้ไขได้",
                value=st.session_state.transcribed_text,
                height=110,
            ).strip()

            if st.button("🔍 ขอสุภาษิต", type="primary", use_container_width=True):
                if not edited_text:
                    st.error("กรุณาใส่ข้อความก่อน")
                else:
                    with st.spinner("กำลังค้นหาสุภาษิต..."):
                        docs = retrieve_proverbs(edited_text, st.session_state.retriever)

                    with st.spinner("กำลังสร้างผลลัพธ์..."):
                        result = llm_structured_proverb(
                            edited_text,
                            docs,
                            st.session_state.client_and_model,
                        )

                    st.session_state.final_result = result

        else:
            st.info("👈 อัดเสียงจากด้านซ้ายก่อน")

    # =====================================================
    # Final result display
    # =====================================================
    if "final_result" in st.session_state:
        st.divider()
        r = st.session_state.final_result

        thai_proverb = (r.get("thai_proverb") or "").strip()
        explanation = (r.get("explanation") or "").strip()

        if not thai_proverb or not explanation:
            st.error("ผลลัพธ์ไม่ครบ")
            st.json(r)
            return

        # Display proverb and explanation
        st.markdown("""
            <style>
            .big-explanation {
            font-size: 20px;
            line-height: 1.7;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown(f"## ✨ {thai_proverb}")

        safe = html.escape(explanation).replace("\n", "<br>")
        st.markdown(f"<div class='big-explanation'>{safe}</div>", unsafe_allow_html=True)

        # TTS button
        if st.button("🔊 อ่านออกมา", use_container_width=True):
            with st.spinner("กำลังสร้างเสียง..."):
                try:
                    audio_bytes = asyncio.run(text_to_speech_edge_tts(explanation))
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("✅ อ่านเสร็จแล้ว")
                except Exception as e:
                    st.error(f"❌ สร้างเสียงไม่สำเร็จ:\n{e}")


if __name__ == "__main__":
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    main()