import streamlit as st
import re
from io import BytesIO
from pypdf import PdfReader
from docx import Document


# ---------------------------------
# 1. KEYWORD EXTRACTION
# ---------------------------------

STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "what", "when", "where", "who", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "this", "that", "it", "from",
    "by", "as", "at", "be", "has", "have", "had"
}


def get_keywords(text):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [word for word in words if word not in STOP_WORDS]


# ---------------------------------
# 2. DOCUMENT EXTRACTION
# ---------------------------------

def extract_document(uploaded_file):

    file_name = uploaded_file.name.lower()
    file_bytes = BytesIO(uploaded_file.getvalue())

    pages = []

    if file_name.endswith(".pdf"):

        pdf = PdfReader(file_bytes)

        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            pages.append({
                "page": page_number,
                "text": text
            })

    elif file_name.endswith(".docx"):

        doc = Document(file_bytes)

        text = "\n".join(
            paragraph.text for paragraph in doc.paragraphs
        )

        pages.append({
            "page": 1,
            "text": text
        })

    elif file_name.endswith(".txt"):

        text = uploaded_file.getvalue().decode(
            "utf-8", errors="ignore"
        )

        pages.append({
            "page": 1,
            "text": text
        })

    return pages


# ---------------------------------
# 3. KNOWLEDGE BASE CREATION
# ---------------------------------

def split_into_facts(pages):

    facts = []
    fact_id = 1

    for page in pages:

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            page["text"]
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) > 10:

                facts.append({
                    "id": fact_id,
                    "page": page["page"],
                    "text": sentence
                })

                fact_id += 1

    return facts


# ---------------------------------
# 4. KNOWLEDGE BASE SEARCH
# ---------------------------------

def search_knowledge_base(question, facts):

    question_words = set(get_keywords(question))
    results = []

    if not question_words:
        return results

    for fact in facts:

        fact_words = set(get_keywords(fact["text"]))

        common = question_words.intersection(fact_words)

        if not common:
            continue

        coverage = len(common) / len(question_words)

        density = len(common) / max(len(fact_words), 1)

        score = (0.75 * coverage) + (0.25 * density)

        meaningful_phrase = " ".join(get_keywords(question))

        if (
            len(meaningful_phrase) > 3
            and meaningful_phrase in fact["text"].lower()
        ):
            score += 0.15

        results.append({
            **fact,
            "score": min(score, 1.0),
            "matched": sorted(common)
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# ---------------------------------
# 5. ANSWER GENERATION
# ---------------------------------

def make_answer(question, results):

    if not results or results[0]["score"] < 0.12:

        return (
            "I could not find a sufficiently relevant answer "
            "in the uploaded document. Try using important "
            "words from the document."
        ), ""

    selected = []

    for result in results:

        if all(
            result["text"].lower() != old["text"].lower()
            for old in selected
        ):
            selected.append(result)

        if len(selected) == 2:
            break

    answer = " ".join(
        item["text"] for item in selected
    )

    sources = ", ".join(
        f"Fact {item['id']} (page {item['page']})"
        for item in selected
    )

    return answer, sources


# ---------------------------------
# 6. SYSTEM EVALUATION
# ---------------------------------

def evaluate_question(question, expected_keywords, facts):

    results = search_knowledge_base(question, facts)

    if not results:
        return False, "No matching fact was found."

    answer_data = make_answer(question, results)

    answer = answer_data[0]

    expected = set(get_keywords(expected_keywords))
    actual = set(get_keywords(answer))

    if not expected:
        return False, "Enter at least one expected keyword."

    matched = expected.intersection(actual)

    coverage = len(matched) / len(expected)

    return (
        coverage >= 0.5,
        f"{len(matched)} of {len(expected)} expected "
        f"keywords matched ({coverage:.0%})."
    )


# ---------------------------------
# 7. STREAMLIT APPLICATION
# ---------------------------------

st.set_page_config(
    page_title="AI Document Question Answering System",
    page_icon="📚",
    layout="wide"
)

st.title("📚AI-Based Document Question Answering System")

st.caption(
    "A rule-based document search and question-answering "
    "application without NLP libraries, embeddings, LLMs or RAG."
)


# SIDEBAR

with st.sidebar:

    st.header("About this project")

    st.write(
        "*Approach:* sentence-based knowledge "
        "representation, keyword matching, rule-based "
        "ranking and extractive answers."
    )

    st.write(
        "*Supported files:* PDF, DOCX and TXT."
    )

    st.info(
        "Answers are taken from the uploaded document. "
        "The system does not use external knowledge."
    )


# DOCUMENT UPLOAD

uploaded_file = st.file_uploader(
    "1. Upload your document",
    type=["pdf", "docx", "txt"]
)


if uploaded_file:

    try:

        pages = extract_document(uploaded_file)

        facts = split_into_facts(pages)

        st.session_state["facts"] = facts

        st.session_state["document_name"] = uploaded_file.name

        if not facts:

            st.error(
                "No readable sentences were extracted. "
                "Try a text-based PDF, DOCX, or TXT file."
            )

        else:

            st.success(
                f"Document processed successfully: "
                f"{len(facts)} facts created from "
                f"{uploaded_file.name}."
            )

            tab_qa, tab_kb, tab_eval, tab_about = st.tabs([
                "💬 Ask Questions",
                "🗂️ Knowledge Base",
                "📊 Evaluation",
                "ℹ️ How It Works"
            ])


            # ---------------------------------
            # QUESTION ANSWERING TAB
            # ---------------------------------

            with tab_qa:

                st.subheader(
                    "2. Ask a question about your document"
                )

                question = st.text_input(
                    "Enter your question",
                    placeholder="Example: What is artificial intelligence?"
                )

                if st.button("Find Answer", type="primary"):

                    if not question.strip():

                        st.warning("Please enter a question.")

                    else:

                        results = search_knowledge_base(
                            question, facts
                        )

                        if results and results[0]["score"] >= 0.12:

                            answer, sources = make_answer(
                                question, results
                            )

                            st.markdown("### Answer")

                            st.write(answer)

                            st.caption(f"Source: {sources}")

                        else:

                            st.warning(
                                make_answer(question, results)[0]
                            )

                        with st.expander(
                            "View search results and matching rules"
                        ):

                            if results:

                                for item in results[:5]:

                                    st.write(
                                        f"**Fact {item['id']} | "
                                        f"Page {item['page']} | "
                                        f"Score: {item['score']:.3f}**"
                                    )

                                    st.write(item["text"])

                                    st.caption(
                                        "Matched keywords: "
                                        + ", ".join(item["matched"])
                                    )

                            else:

                                st.write("No matching facts.")


            # ---------------------------------
            # KNOWLEDGE BASE TAB
            # ---------------------------------

            with tab_kb:

                st.subheader(
                    "3. Constructed Knowledge Base"
                )

                st.write(
                    "Each extracted sentence is stored "
                    "as a fact with an ID and source page."
                )

                st.dataframe(
                    [
                        {
                            "Fact ID": f["id"],
                            "Page": f["page"],
                            "Knowledge / Fact": f["text"]
                        }
                        for f in facts
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = (
                    "Fact ID,Page,Knowledge / Fact\n"
                    + "\n".join(
                        f'"{f["id"]}","{f["page"]}",'
                        f'"{f["text"].replace(chr(34), chr(34)*2)}"'
                        for f in facts
                    )
                )

                st.download_button(
                    "Download knowledge base as CSV",
                    data=csv_data,
                    file_name="knowledge_base.csv",
                    mime="text/csv"
                )


            # ---------------------------------
            # EVALUATION TAB
            # ---------------------------------

            with tab_eval:

                st.subheader("4. Evaluate the system")

                st.write(
                    "Enter a test question and the important "
                    "keywords expected in a correct answer."
                )

                eval_question = st.text_input(
                    "Test question",
                    key="eval_question"
                )

                expected = st.text_input(
                    "Expected answer keywords",
                    key="expected_keywords",
                    placeholder="Example: machine intelligence learning"
                )

                if st.button("Evaluate Test Question"):

                    if (
                        not eval_question.strip()
                        or not expected.strip()
                    ):

                        st.warning(
                            "Enter both a question and expected keywords."
                        )

                    else:

                        passed, message = evaluate_question(
                            eval_question,
                            expected,
                            facts
                        )

                        if passed:

                            st.success(
                                "Test passed: " + message
                            )

                        else:

                            st.warning(
                                "Test not passed: " + message
                            )

                st.markdown(
                    "*Suggested evaluation criteria*"
                )

                st.write(
                    "- Relevance: does the returned fact relate to the question?"
                )

                st.write(
                    "- Correctness: is the answer supported by the uploaded document?"
                )

                st.write(
                    "- Source traceability: does the answer show its fact ID and page?"
                )

                st.write(
                    "- Keyword coverage: what fraction of expected keywords appear in the answer?"
                )


            # ---------------------------------
            # HOW IT WORKS TAB
            # ---------------------------------

            with tab_about:

                st.subheader("System methodology")

                st.markdown("""
                1. *Document processing:* extracts text from PDF, DOCX or TXT files.

                2. *Knowledge representation:* splits extracted text into sentence-level facts. Each fact stores an ID, sentence and page number.

                3. *Question processing:* converts the question into lowercase word tokens and removes a fixed list of common words.

                4. *Search and reasoning:* compares question keywords with each fact. Facts are ranked using keyword coverage and keyword density.

                5. *Answer generation:* returns the highest-ranked sentence(s) directly from the document and displays their sources.

                6. *Evaluation:* compares expected keywords with answer keywords and reports keyword coverage.

                *Why this approach?* It is transparent, lightweight and rule-based. It avoids NLP toolkits, machine-learning language models, embeddings and RAG.
                """)


    except Exception as error:

        st.error(
            f"Could not process this file: {error}"
        )


else:

    st.info(
        "Upload a document to begin. For your first test, "
        "use a small text-based PDF or DOCX."
    )

    st.markdown("### Quick demonstration")

    st.write(
        "After uploading a document, ask questions using "
        "words that appear in the document, such as "
        "'What is the definition of ...?'"
    )