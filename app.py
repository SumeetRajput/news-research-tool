"""
app.py
------
The INTERFACE layer. Run with:  streamlit run app.py

Three things beyond a plain form:

* Voice in   - record a question with the microphone, Whisper transcribes it
* Voice out  - Orpheus reads the answer back
* Take-away  - download the whole session as PDF, Markdown, or a CSV of sources

All of it runs on one provider (Groq), so there is a single key to manage.
"""

from __future__ import annotations

import streamlit as st

from audio_io import TTS_VOICES, AudioError, synthesize, transcribe
from langchain_config import (
    ConfigError,
    DEFAULT_MODEL,
    build_quick_summary_chain,
    is_configured,
    list_models,
    test_connection,
)
from news_client import NewsClientError, load_documents, search_articles
from rag_engine import answer_question, build_index
from report import safe_filename, sources_to_csv, to_markdown, to_pdf

st.set_page_config(
    page_title="Equity Research News Tool",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      h1, h2, h3 { letter-spacing: -0.015em; }

      .hero {
        border-radius: 14px;
        padding: 1.6rem 1.9rem;
        background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(14,165,233,.10));
        border: 1px solid rgba(125,125,155,.22);
        margin-bottom: 1.4rem;
      }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 1.95rem; }
      .hero p  { margin: 0; opacity: .78; font-size: .95rem; }

      .answer {
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(125,125,155,.22);
        border-left: 3px solid #6366f1;
        background: rgba(125,125,155,.06);
        line-height: 1.65;
      }
      .src {
        border-radius: 9px;
        padding: .6rem .85rem;
        margin-bottom: .45rem;
        border: 1px solid rgba(125,125,155,.18);
        background: rgba(125,125,155,.05);
        font-size: .87rem;
      }
      .src .ref { color: #6366f1; font-weight: 700; margin-right: .35rem; }
      .src .meta { opacity: .6; font-size: .78rem; }

      .pill {
        display: inline-block; padding: .16rem .6rem; border-radius: 999px;
        font-size: .72rem; font-weight: 600; margin-right: .35rem;
        background: rgba(99,102,241,.16); border: 1px solid rgba(99,102,241,.3);
      }
      .stButton>button { border-radius: 9px; font-weight: 600; }
      div[data-testid="stDownloadButton"]>button { border-radius: 9px; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
_DEFAULTS = {
    "vectorstore": None,
    "indexed_topic": None,
    "article_count": 0,
    "full_text_count": 0,
    "entries": [],          # every Q&A this session, used for the exports
    "model": DEFAULT_MODEL,
    "conn_result": None,
    "pending_question": "",
    "last_audio_id": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

ready, ready_reason = is_configured()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Groq")
    st.caption("Reasoning, speech-to-text and speech synthesis all run here.")

    if ready:
        st.success(ready_reason, icon="✅")
        models = list_models()
        idx = models.index(st.session_state.model) if st.session_state.model in models else 0
        st.session_state.model = st.selectbox("Reasoning model", models, index=idx)
    else:
        st.error(ready_reason, icon="⚠️")
        st.markdown("[Get a free key →](https://console.groq.com/keys)")

    if st.button("Test connection", use_container_width=True):
        with st.spinner("Pinging Groq…"):
            st.session_state.conn_result = test_connection(st.session_state.model)
    if st.session_state.conn_result:
        ok, msg = st.session_state.conn_result
        (st.success if ok else st.error)(msg)

    st.divider()
    st.markdown("### 🔊 Voice")
    voice_enabled = st.toggle("Read answers aloud", value=True)
    voice = st.selectbox("Voice", TTS_VOICES, index=0, disabled=not voice_enabled)

    st.divider()
    st.markdown("### 🔎 Search")
    mode = st.radio(
        "Mode",
        ["Deep research", "Quick briefing"],
        captions=["Reads full articles, cites sources", "Headlines only, fast and shallow"],
    )
    days_back = st.slider("Look back (days)", 1, 28, 14)
    max_articles = st.slider("Articles to pull", 3, 20, 10)
    top_k = st.slider("Passages per answer", 3, 12, 6)

    st.divider()
    st.caption("Embeddings run locally and are always free.")

model = st.session_state.model

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📰 Equity Research News Tool</h1>
      <p>Ask by voice or text. It finds recent coverage, reads it, and answers with sources you can trace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ready:
    st.warning("Add your Groq key to `.env` and restart to begin.")
    st.stop()

# --------------------------------------------------------------------------
# Topic + index building
# --------------------------------------------------------------------------
col_topic, col_go = st.columns([4, 1], vertical_alignment="bottom")
with col_topic:
    topic = st.text_input("Topic or company", placeholder="e.g. Tata Motors EV strategy")
with col_go:
    go = st.button(
        "Build index" if mode == "Deep research" else "Get briefing",
        type="primary",
        use_container_width=True,
    )


def render_sources(sources: list[dict]) -> None:
    for s in sources:
        st.markdown(
            f"<div class='src'><span class='ref'>[{s['n']}]</span>"
            f"<a href='{s['url']}' target='_blank'>{s['title']}</a>"
            f"<div class='meta'>{s['source']} · {s['published_at']}</div></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Quick briefing
# --------------------------------------------------------------------------
if mode == "Quick briefing":
    if go:
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            try:
                with st.spinner("Searching news…"):
                    articles = search_articles(topic, days_back, max_articles)
                if not articles:
                    st.info("No articles found. Try broader wording or a longer window.")
                else:
                    headlines = "\n".join(
                        f"- {a['title']} ({a['source']}, {a['published_at']}): {a['blurb']}"
                        for a in articles
                    )
                    with st.spinner("Summarising…"):
                        summary = build_quick_summary_chain(model).invoke(
                            {"query": topic, "headlines": headlines}
                        )
                    st.markdown(f"<div class='answer'>{summary}</div>", unsafe_allow_html=True)
                    if voice_enabled:
                        try:
                            st.audio(synthesize(summary, voice), format="audio/wav")
                        except AudioError as exc:
                            st.caption(f"Audio unavailable: {exc}")
                    with st.expander(f"Headlines used ({len(articles)})"):
                        for a in articles:
                            st.markdown(f"- [{a['title']}]({a['url']}) — {a['source']}, {a['published_at']}")
            except (NewsClientError, ConfigError) as exc:
                st.error(str(exc))

# --------------------------------------------------------------------------
# Deep research
# --------------------------------------------------------------------------
else:
    if go:
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            try:
                with st.status("Researching…", expanded=True) as status:
                    st.write("Finding articles")
                    articles = search_articles(topic, days_back, max_articles)
                    if not articles:
                        status.update(label="No articles found", state="error")
                        st.stop()

                    st.write(f"Reading {len(articles)} articles")
                    docs = load_documents(articles)
                    if not docs:
                        status.update(label="Everything was paywalled", state="error")
                        st.stop()

                    st.write("Embedding and indexing")
                    st.session_state.vectorstore = build_index(docs)
                    st.session_state.indexed_topic = topic
                    st.session_state.article_count = len(docs)
                    st.session_state.full_text_count = sum(
                        1 for d in docs if d.metadata.get("full_text")
                    )
                    st.session_state.entries = []
                    status.update(label=f"Indexed {len(docs)} articles", state="complete")
            except (NewsClientError, ConfigError, ValueError) as exc:
                st.error(str(exc))

    if st.session_state.vectorstore is not None:
        a, b, c = st.columns(3)
        a.metric("Topic", st.session_state.indexed_topic)
        b.metric("Articles indexed", st.session_state.article_count)
        c.metric("With full text", st.session_state.full_text_count)

        st.divider()
        st.markdown("#### Ask a question")

        tab_type, tab_speak = st.tabs(["⌨️  Type", "🎙️  Speak"])

        with tab_type:
            typed = st.text_input(
                "Question",
                value=st.session_state.pending_question,
                placeholder="e.g. What are analysts saying about margin pressure?",
                label_visibility="collapsed",
            )
            ask_typed = st.button("Ask", type="primary")

        with tab_speak:
            st.caption("Record your question, then release. Whisper transcribes it on Groq.")
            clip = st.audio_input("Record", label_visibility="collapsed")
            if clip is not None and id(clip) != st.session_state.last_audio_id:
                st.session_state.last_audio_id = id(clip)
                try:
                    with st.spinner("Transcribing…"):
                        heard = transcribe(clip.getvalue())
                    if heard:
                        st.session_state.pending_question = heard
                        st.success(f"Heard: “{heard}”")
                        st.rerun()
                    else:
                        st.warning("Didn't catch that — try again.")
                except AudioError as exc:
                    st.error(str(exc))

        question = (typed or st.session_state.pending_question).strip()

        if ask_typed and question:
            try:
                with st.spinner("Retrieving and reasoning…"):
                    result = answer_question(
                        question, st.session_state.vectorstore, k=top_k, model=model
                    )
                st.session_state.entries.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )
                st.session_state.pending_question = ""
            except ConfigError as exc:
                st.error(str(exc))

        # ------------------------------------------------------------------
        # Results, newest first
        # ------------------------------------------------------------------
        if st.session_state.entries:
            st.divider()
            for i, entry in enumerate(reversed(st.session_state.entries)):
                st.markdown(f"##### {entry['question']}")
                st.markdown(f"<div class='answer'>{entry['answer']}</div>", unsafe_allow_html=True)

                if voice_enabled:
                    with st.spinner("Generating audio…"):
                        try:
                            st.audio(synthesize(entry["answer"], voice), format="audio/wav")
                        except AudioError as exc:
                            st.caption(f"Audio unavailable: {exc}")

                if entry["sources"]:
                    with st.expander(f"Sources ({len(entry['sources'])})", expanded=(i == 0)):
                        render_sources(entry["sources"])
                st.write("")

            # --------------------------------------------------------------
            # Downloads
            # --------------------------------------------------------------
            st.divider()
            st.markdown("#### 📥 Take it with you")
            stem = safe_filename(st.session_state.indexed_topic)
            entries = st.session_state.entries
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "PDF brief",
                to_pdf(st.session_state.indexed_topic, entries),
                f"{stem}.pdf",
                "application/pdf",
            )
            d2.download_button(
                "Markdown",
                to_markdown(st.session_state.indexed_topic, entries),
                f"{stem}.md",
                "text/markdown",
            )
            d3.download_button(
                "Sources CSV",
                sources_to_csv(entries),
                f"{stem}-sources.csv",
                "text/csv",
            )
            st.caption(
                f"{len(entries)} question{'s' if len(entries) != 1 else ''} in this session. "
                "Exports include every answer and its citations."
            )
    else:
        st.info("Enter a topic above and build an index to start asking questions.")
