import tempfile
from pathlib import Path

import streamlit as st
from app.qa import ask
from ingest.embed_and_index import upload_files

def sources_snippet(sources):
    snippets = []
    for s in sources:
        loc = f"p.{s['page']}" if s.get("page") else (f"slide {s['slide']}" if s.get("slide") else "")
        snippet_preview = (s.get("snippet") or s.get("text") or "")[:80]
        snippets.append(f"- **{s['title']}** {loc} — `{s['source']}`  \n{snippet_preview}…")
    return snippets

st.markdown("""
<style>
/* The button of the sliders */
.stSlider [role="slider"] {
    background-color: #0072C6;
}

/* Center the app title */
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("IBM x DeepSea Pythia RAG Demo")

# --- session state init ---
if "history" not in st.session_state: st.session_state.history = []
if "temperature" not in st.session_state: st.session_state.temperature = 0.0
if "top_k" not in st.session_state: st.session_state.top_k = 6
if "upload_feedback" not in st.session_state: st.session_state.upload_feedback = None
if "clear_upload_docs" not in st.session_state: st.session_state.clear_upload_docs = False
if "uploader_version" not in st.session_state: st.session_state.uploader_version = 0

if st.session_state.get("clear_upload_docs"):
    st.session_state.pop(f"upload_docs_{st.session_state.uploader_version}", None)
    st.session_state.clear_upload_docs = False

if st.session_state.upload_feedback:
    level, message = st.session_state.upload_feedback
    getattr(st, level)(message)
    st.session_state.upload_feedback = None


# --- Popup definition ---
@st.dialog("Upload Files", width="medium", dismissible=True)
def upload_dialog():
    st.write("Upload one or more documents to add them to your retrieval index.")

    uploader_key = f"upload_docs_{st.session_state.uploader_version}"
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "pptx", "txt", "md", "docx", "mp4"],
        accept_multiple_files=True,
        key=uploader_key
    )

    col_ingest, col_cancel = st.columns([2,1])

    with col_ingest:
        if st.button("Ingest", key="ingest_files", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one file to ingest.")
            else:
                tmp_paths = []
                total = len(uploaded_files)
                print(f"[main] Ingesting {total} files...")

                progress_label = st.empty()
                progress_bar = st.progress(0.0)

                try:
                    for i, file in enumerate(uploaded_files, start=1):
                        # Save file to temp (simple one-shot write)
                        tmp_dir = Path(tempfile.gettempdir())
                        tmp_path = tmp_dir / file.name
                        with open(tmp_path, "wb") as tmp:
                            tmp.write(file.getbuffer())

                        progress_label.text(f"Indexing document {file.name}…")
                        upload_files([str(tmp_path)], max_tokens=512, overlap_tokens=120)

                        # Step the bar by 1/N after each file finishes
                        progress_bar.progress(i / total)
                        tmp_paths.append(str(tmp_path))                    
                    progress_bar.progress(1.0)
                    progress_label.text("Done.")
                    st.session_state.upload_feedback = ("success", "Upload completed and documents indexed.")

                except Exception as exc:
                    st.session_state.upload_feedback = ("error", f"Failed to ingest files: {exc}")
                finally:
                    for path in tmp_paths:
                        try:
                            Path(path).unlink()
                        except OSError:
                            pass

                    st.session_state.uploader_version += 1
                    st.session_state.clear_upload_docs = True
                    st.rerun()  # closes dialog

    with col_cancel:
        if st.button("Cancel", key="cancel_upload", use_container_width=True):
            st.session_state.uploader_version += 1
            st.session_state.clear_upload_docs = True
            st.rerun()  # closes dialog

# --- Sidebar controls + button to open dialog ---
with st.sidebar:
    st.header("Controls")
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.01,
        help="Higher = more diverse/creative sampling; lower = more deterministic."
    )
    st.session_state.top_k = st.slider(
        "Top-K answers",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k,
        step=1,
        help="How many chunks/answers to retrieve from your index."
    )
    
    col1, col2, col3 = st.columns([1,4,1])
    if col2.button("New Chat", use_container_width=True):
        st.session_state.history = []
        st.success("New chat started.")
        st.rerun()

    st.divider()

    # Button to open upload dialog
    if st.button("Upload Files…", use_container_width=True):
        upload_dialog()

# --- Chat interface ---
q = st.chat_input("Ask about your docs…")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        snippets = sources_snippet(turn["sources"])
        if snippets:
            with st.expander("Sources"):
                for s in snippets:
                    st.markdown(s)

if q:
    with st.chat_message("user"):
        st.write(q)
    k = st.session_state.top_k
    temperature = st.session_state.temperature
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with message_placeholder.container():
            with st.spinner("Fetching relevant chunks and composing an answer..."):
                result = ask(q, k=k, temperature=temperature)
        st.session_state.history.append({"q": q, **result})
        message_placeholder.empty()
        message_placeholder.write(result["answer"])
        snippets = sources_snippet(result["sources"])
        if snippets:
            with st.expander("Sources"):
                for s in snippets:
                    st.markdown(s)
