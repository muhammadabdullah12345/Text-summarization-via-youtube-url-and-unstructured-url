import validators
import streamlit as st
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

from youtube_transcript_api import YouTubeTranscriptApi


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:
        try:
            with st.spinner("Waiting..."):

                # ✅ LLM
                llm = ChatGroq(
                    model="qwen/qwen3-32b",
                    groq_api_key=groq_api_key,
                    temperature=0.5,
                    max_tokens=512,
                )

                # ✅ LOAD DATA (YouTube + fallback)
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(

                            generic_url,
                            add_video_info=True,
                            language=["en", "en-US"],
                            translation="en",   # 🔥 THIS IS KEY
                        )
                        docs = loader.load()

                    except Exception:
                        st.warning("Primary YouTube loader failed. Trying fallback...")

                        try:
                            # Extract video ID
                            video_id = re.findall(r"v=([^&]+)", generic_url)
                            if not video_id:
                                video_id = re.findall(r"youtu\.be/([^?]+)", generic_url)

                            video_id = video_id[0]

                            transcript = YouTubeTranscriptApi.get_transcript(video_id)
                            text = " ".join([t["text"] for t in transcript])

                            docs = [{"page_content": text}]

                        except Exception:
                            st.error("❌ This video has no transcript or is restricted.")
                            st.stop()

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                # ✅ Combine docs
                if isinstance(docs[0], dict):
                    text = docs[0]["page_content"]
                else:
                    text = " ".join([doc.page_content for doc in docs])

                # ✅ LCEL Chain (modern LangChain)
                chain = prompt | llm | StrOutputParser()

                output_summary = chain.invoke({"text": text})

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")