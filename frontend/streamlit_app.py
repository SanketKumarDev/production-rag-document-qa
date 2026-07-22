import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AskMyTechDocs", page_icon="📚", layout="wide")
st.title("📚 AskMyTechDocs")
st.caption("BM25 + Vector Search + RRF + Cross-Encoder Reranking")

question = st.text_area(
    "Ask a question about your documents",
    placeholder="How does JWT authentication work?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching and generating..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": 5},
            timeout=180,
        )

    if response.ok:
        data = response.json()
        st.subheader("Answer")
        st.write(data["answer"])

        if data["grounded"]:
            st.success("Citation validation passed.")
        else:
            st.warning("Answer was not sufficiently grounded.")

        st.subheader("Citations")
        for citation in data["citations"]:
            page = (
                f" — Page {citation['page']}"
                if citation["page"] is not None
                else ""
            )
            st.write(f"**[{citation['id']}]** {citation['source']}{page}")

        with st.expander("Retrieved evidence"):
            for source in data["retrieved_sources"]:
                st.write(
                    f"**{source['source']}** | "
                    f"score={source['score']:.4f} | "
                    f"chunk={source['chunk_id']}"
                )
    else:
        st.error(response.text)
