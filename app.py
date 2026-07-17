import os
import re
import streamlit as st
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from dotenv import load_dotenv
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator


load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb.utils import embedding_functions

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Archive & Ledger — The Film Society Reading Room",
    page_icon="📇",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  —  "Archive & Ledger" — a film archive card catalog / Criterion
#  essay booklet aesthetic. Paper, ink, rust stamps, hairline rules.
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper:     #F7F3EA;
    --paper-2:   #F1ECDF;
    --ink:       #2B2620;
    --ink-soft:  #5B5346;
    --rust:      #A8492F;
    --sage:      #7A8B6F;
    --hairline:  #C9C2B4;
    --font-display: 'Source Serif 4', Georgia, serif;
    --font-body:    'IBM Plex Sans', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
}
/* ── Chrome hiding ── */
#MainMenu            { visibility: hidden; }
footer               { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── App background: paper, faint hairline grid like a ledger page ── */
.stApp {
    background-color: var(--paper);
    background-image:
        linear-gradient(var(--hairline) 1px, transparent 1px),
        linear-gradient(90deg, var(--hairline) 1px, transparent 1px);
    background-size: 100% 2.1rem, 2.1rem 100%;
    background-position: 0 5.2rem, 0 0;
    background-attachment: fixed;
    opacity: 1;
}
body, .stApp, [class*="css"] { color: var(--ink); font-family: var(--font-body); }

/* ── Main container ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 820px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar        { width: 6px; }
::-webkit-scrollbar-track  { background: var(--paper-2); }
::-webkit-scrollbar-thumb  { background: var(--hairline); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--rust); }

/* ── Top page masthead ── */
.ledger-masthead {
    text-align: center;
    padding: 0.25rem 0 1.4rem 0;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--ink);
}
.ledger-masthead .kicker {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--rust);
    margin-bottom: 0.35rem;
}
.ledger-masthead .title {
    font-family: var(--font-display);
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--ink);
}
.ledger-masthead .subtitle {
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--ink-soft);
    font-style: italic;
    margin-top: 0.25rem;
}

/* ════════════════════════════════════════════════════════════════════════════
   CHAT MESSAGES  —  styled as catalog / index cards
   ════════════════════════════════════════════════════════════════════════════ */

/* — Avatars: square off, quiet tone — */
.stChatMessage [data-testid="stChatMessageAvatarUser"],
.stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 0 !important;
    background: var(--ink) !important;
}

/* — User note (typed request slip) — */
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown {
    background: var(--paper-2);
    border: 1px solid var(--hairline);
    border-left: 3px solid var(--sage);
    border-radius: 0;
    padding: 0.7rem 1.1rem;
    color: var(--ink);
    margin: 0.25rem 0;
    font-family: var(--font-body);
}

/* — User note (typed request slip) — */
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown,
.stChatMessage[data-testid="stChatMessageUser"] .stMarkdown * {
    color: var(--ink) !important;
}

/* — Assistant bubble: the index card itself — */
.stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown {
    background: var(--paper);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    margin: 0.35rem 0 0.9rem 0;
    color: var(--ink);
    line-height: 1.7;
    font-family: var(--font-body);
    position: relative;
    box-shadow: 3px 3px 0 var(--hairline);
}

/* Catch-all: Streamlit's dark theme sets light text colors on inner
   paragraph/list/span nodes, sometimes via emotion-generated rules that
   are re-injected after ours. Repeat the same selector fragment to force
   our specificity above theirs (a plain, reliable override trick),
   target both the ".stMarkdown" wrapper AND the inner
   "stMarkdownContainer" testid Streamlit actually renders text into,
   and force color on every descendant with a wildcard as a final safety
   net — accent colors for strong/em/code are re-applied further below
   and win the tie because they're declared later in this stylesheet. */
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown p,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown li,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown span,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown div,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown td,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown th,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown h2,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown h3,
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown h4,
.stChatMessage[data-testid="stChatMessageAssistant"] [data-testid="stMarkdownContainer"],
.stChatMessage[data-testid="stChatMessageAssistant"] [data-testid="stMarkdownContainer"] *:not(strong):not(em):not(code):not(.accession-tag):not(.accession-tag *),
.stChatMessage[data-testid="stChatMessageAssistant"] .stMarkdown *:not(strong):not(em):not(code):not(.accession-tag):not(.accession-tag *) {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

.stChatMessage[data-testid="stChatMessageUser"] [data-testid="stMarkdownContainer"],
.stChatMessage[data-testid="stChatMessageUser"] [data-testid="stMarkdownContainer"] * {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

/* accession number tag, upper-right corner of every card */
.accession-tag {
    display: inline-block;
    float: right;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--rust);
    border: 1px solid var(--rust);
    padding: 0.12rem 0.5rem;
    margin: -0.2rem -0.1rem 0.6rem 0.8rem;
    background: var(--paper-2);
}

/* ════════════════════════════════════════════════════════════════════════════
   MARKDOWN TYPOGRAPHY  (inside assistant bubbles only)
   ════════════════════════════════════════════════════════════════════════════ */

.stChatMessage[data-testid="stChatMessageAssistant"] h2 {
    font-family: var(--font-display) !important;
    color: var(--ink) !important;
    font-size: 1.4rem !important;
    font-weight: 600  !important;
    margin-top: 1.4rem  !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 2px solid var(--ink) !important;
}

/* h3 = single-movie title header, styled like a catalog card headword,
   with a rotated rust rubber-stamp badge beside it */
.stChatMessage[data-testid="stChatMessageAssistant"] h3 {
    font-family: var(--font-display) !important;
    font-style: italic !important;
    color: var(--ink) !important;
    font-size: 1.25rem !important;
    font-weight: 600  !important;
    margin-top: 1.1rem  !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 0.3rem !important;
    border-bottom: 1px solid var(--hairline) !important;
    position: relative;
}
.stChatMessage[data-testid="stChatMessageAssistant"] h3::after,
.stChatMessage[data-testid="stChatMessageAssistant"] h4::after {
    content: "ARCHIVED";
    display: inline-block;
    font-family: var(--font-mono);
    font-style: normal;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--rust);
    border: 1.5px solid var(--rust);
    padding: 0.1rem 0.4rem;
    margin-left: 0.65rem;
    transform: rotate(-4deg);
    vertical-align: middle;
    opacity: 0.9;
}

/* h4 = per-movie entry heading in recommendation lists */
.stChatMessage[data-testid="stChatMessageAssistant"] h4 {
    font-family: var(--font-display) !important;
    color: var(--ink) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.3rem !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] strong,
.stChatMessage[data-testid="stChatMessageAssistant"] [data-testid="stMarkdownContainer"] strong {
    color: var(--rust) !important;
    -webkit-text-fill-color: var(--rust) !important;
    font-weight: 600  !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] em,
.stChatMessage[data-testid="stChatMessageAssistant"] [data-testid="stMarkdownContainer"] em {
    color: var(--ink-soft) !important;
    -webkit-text-fill-color: var(--ink-soft) !important;
    font-family: var(--font-mono);
    font-style: normal;
    font-size: 0.85em;
}

.stChatMessage[data-testid="stChatMessageAssistant"] p {
    margin-bottom: 0.55rem !important;
}

/* — Lists — */
.stChatMessage[data-testid="stChatMessageAssistant"] ul,
.stChatMessage[data-testid="stChatMessageAssistant"] ol {
    padding-left: 1.5rem  !important;
    margin-top: 0.25rem   !important;
    margin-bottom: 0.55rem !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] li {
    margin-bottom: 0.2rem !important;
    color: var(--ink) !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] li::marker {
    color: var(--rust) !important;
}

/* — Horizontal rules (key section separator) — a torn perforation line — */
.stChatMessage[data-testid="stChatMessageAssistant"] hr {
    border: none               !important;
    border-top: 1px dashed var(--hairline) !important;
    margin: 1.2rem 0           !important;
    opacity: 1;
}

/* — Inline code — */
.stChatMessage[data-testid="stChatMessageAssistant"].stChatMessage[data-testid="stChatMessageAssistant"] code,
.stChatMessage[data-testid="stChatMessageAssistant"] [data-testid="stMarkdownContainer"] code {
    background: var(--paper-2) !important;
    border: 1px solid var(--hairline) !important;
    color: var(--rust)    !important;
    -webkit-text-fill-color: var(--rust) !important;
    padding: 0.1rem 0.4rem !important;
    border-radius: 0     !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82em      !important;
}

/* — Blockquote (used for overviews) — */
.stChatMessage[data-testid="stChatMessageAssistant"] blockquote {
    border-left: 3px solid var(--sage) !important;
    padding: 0.5rem 1rem           !important;
    margin: 0.5rem 0               !important;
    background: var(--paper-2)     !important;
    border-radius: 0                !important;
    color: var(--ink-soft)         !important;
    font-style: italic             !important;
}

/* — Tables (catalog record fields) — */
.stChatMessage[data-testid="stChatMessageAssistant"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0 !important;
    font-size: 0.9rem;
    border: 1px solid var(--hairline) !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] thead th {
    background: var(--paper-2) !important;
    color: var(--ink)     !important;
    text-align: left    !important;
    padding: 0.5rem 0.75rem !important;
    border-bottom: 2px solid var(--ink) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 600    !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] tbody td {
    padding: 0.4rem 0.75rem !important;
    border-bottom: 1px solid var(--hairline) !important;
    color: var(--ink)          !important;
}
.stChatMessage[data-testid="stChatMessageAssistant"] tbody td:first-child {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--ink-soft) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   EXPANDERS  →  "Catalog Record" pull-tab (source panels inside index cards)
   ════════════════════════════════════════════════════════════════════════════ */
.stChatMessage[data-testid="stChatMessageAssistant"] .stExpander {
    border: 1px dashed var(--rust) !important;
    border-radius: 0       !important;
    background: var(--paper-2)      !important;
    margin-top: 0.85rem      !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] .stExpander summary {
    background: transparent  !important;
    color: var(--rust)           !important;
    font-family: var(--font-mono) !important;
    font-size: 0.76rem       !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

.stChatMessage[data-testid="stChatMessageAssistant"] .stExpander .stMarkdown {
    background: transparent !important;
    border: none            !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.82rem      !important;
    color: var(--ink-soft)          !important;
    line-height: 1.5        !important;
    font-family: var(--font-body) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   SIDEBAR  —  the card-catalog drawer
   ════════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--paper-2) !important;
    border-right: 1px solid var(--ink);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--ink-soft) !important;
    font-family: var(--font-body);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--ink) !important;
    font-family: var(--font-display) !important;
}

[data-testid="stSidebar"] h3 {
    font-size: 0.8rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: var(--font-mono) !important;
    color: var(--rust) !important;
    border-bottom: 1px solid var(--hairline);
    padding-bottom: 0.3rem;
    margin-top: 0.5rem;
}

[data-testid="stSidebar"] hr {
    border-top: 1px dashed var(--hairline) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: var(--paper)           !important;
    border: 1px solid var(--ink)    !important;
    color: var(--ink)               !important;
    border-radius: 0           !important;
    width: 100%;
    transition: all 0.15s ease;
    font-weight: 500;
    font-family: var(--font-body);
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--rust)       !important;
    border-color: var(--rust)     !important;
    color: var(--paper)            !important;
}

/* — Sidebar open/close (collapse) control — darken the icon so it's
   visible against the light theme, in both its collapsed and
   expanded-sidebar positions. Streamlit's exact data-testid for this
   button has changed across versions, so match generously via substring
   attribute selectors ([data-testid*="..."]) on top of the exact names. */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid*="ollapse"],
[data-testid*="ollapsedControl"],
header[data-testid="stHeader"] button,
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] button[kind="headerNoPadding"] {
    color: var(--ink) !important;
}

[data-testid="stSidebarCollapseButton"] *,
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid*="ollapse"] *,
[data-testid*="ollapsedControl"] *,
header[data-testid="stHeader"] button *,
[data-testid="stSidebar"] button[kind="header"] *,
[data-testid="stSidebar"] button[kind="headerNoPadding"] *,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid*="ollapse"] svg,
header[data-testid="stHeader"] svg,
[data-testid="baseButton-header"] svg,
[data-testid="baseButton-headerNoPadding"] svg {
    fill: var(--ink) !important;
    stroke: var(--ink) !important;
    color: var(--ink) !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button,
header[data-testid="stHeader"] button,
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] button[kind="headerNoPadding"] {
    background: var(--paper-2) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   CHAT INPUT  —  a ruled ledger line
   ════════════════════════════════════════════════════════════════════════════ */
.stChatInputContainer {
    border-radius: 0       !important;
    border: 1px solid var(--ink) !important;
    background: var(--paper)       !important;
}

.stChatInputContainer textarea {
    color: var(--ink)      !important;
    background: transparent !important;
    font-family: var(--font-body) !important;
}

.stChatInputContainer textarea::placeholder {
    color: var(--ink-soft) !important;
    font-style: italic;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--rust) !important; }
.stSpinner p { font-family: var(--font-mono) !important; color: var(--ink-soft) !important; font-size: 0.82rem !important; }

/* ════════════════════════════════════════════════════════════════════════════
   WELCOME CARD  —  the reading room's opening index card
   ════════════════════════════════════════════════════════════════════════════ */
.welcome-card {
    background: var(--paper);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 2.6rem 2rem;
    text-align: center;
    margin: 0.5rem 0 1.5rem 0;
    box-shadow: 4px 4px 0 var(--hairline);
    position: relative;
}

.welcome-icon {
    font-size: 2.4rem;
    margin-bottom: 0.6rem;
    display: inline-block;
}

.welcome-title {
    color: var(--ink) !important;
    font-family: var(--font-display) !important;
    font-style: italic;
    font-size: 1.75rem !important;
    font-weight: 600  !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.01em;
}

.welcome-sub {
    color: var(--ink-soft) !important;
    font-size: 0.95rem !important;
    line-height: 1.65  !important;
    max-width: 420px;
    margin: 0 auto !important;
    text-align: center !important;
}

.welcome-card p,
.welcome-card div {
    text-align: center !important;
}

/* ── Suggestion chips (as catalog request slips) ──
   Broad + specific selectors so this beats Streamlit's dark-theme button
   defaults (background + text color are set on the button AND its inner
   paragraph/markdown wrapper). */
div[data-testid="stAppViewContainer"] .stButton > button,
[data-testid="column"] .stButton > button,
.main .stButton > button {
    background: var(--paper-2) !important;
    border: 1px solid var(--hairline) !important;
    border-left: 3px solid var(--sage) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    padding: 0.6rem 0.9rem !important;
    transition: all 0.15s ease;
}

div[data-testid="stAppViewContainer"] .stButton > button *,
[data-testid="column"] .stButton > button *,
.main .stButton > button * {
    color: var(--ink) !important;
}

div[data-testid="stAppViewContainer"] .stButton > button:hover,
[data-testid="column"] .stButton > button:hover,
.main .stButton > button:hover {
    background: var(--paper) !important;
    border-color: var(--rust) !important;
    border-left-color: var(--rust) !important;
    color: var(--rust) !important;
}

div[data-testid="stAppViewContainer"] .stButton > button:hover *,
[data-testid="column"] .stButton > button:hover *,
.main .stButton > button:hover * {
    color: var(--rust) !important;
}

/* ── Source chunk cards (inside Catalog Record pull-tab) ── */
.source-chunk {
    background: var(--paper);
    border: 1px solid var(--hairline);
    border-radius: 0;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.45rem;
    font-size: 0.8rem;
    color: var(--ink-soft);
    line-height: 1.55;
    font-family: var(--font-body);
}

.source-chunk strong {
    color: var(--rust) !important;
    font-size: 0.75rem !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Sidebar footer ── */
.sidebar-footer {
    text-align: center;
    color: var(--ink-soft);
    font-size: 0.68rem;
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
    margin-top: 1.5rem;
    padding-top: 0.75rem;
    border-top: 1px dashed var(--hairline);
}
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD RAG CHAIN  (cached — built only once)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,max_tokens=1024)

    # ── Improved prompt with strict formatting rules ──────────────────────────
    template = """You are a knowledgeable and concise movie assistant. Use ONLY the movie details provided in the context below to answer the question.

The context contains structured movie entries with fields: Title, Year, Director, Genres, Rating, Cast (Actor as Character), and Overview.

## FORMATTING RULES — follow these EXACTLY:

### When asked about a SINGLE movie:

---

### Title (Year)

| Field      | Details |
|------------|---------|
| **Director** | ... |
| **Genres**   | ... |
| **Rating**   | ... |

**Cast:**
- Actor as Character
- Actor as Character

**Overview:**
> Write the overview text inside a blockquote like this.

---

### When asked for a LIST / RECOMMENDATION (e.g. "suggest horror movies"):

For EACH movie use this compact format, separated by `---`:

---

#### 1. Title (Year) — ⭐ Rating
*Genres:* Genre1, Genre2 · *Director:* Name

> Short overview here.

---

### Universal rules:
1. ALWAYS use `---` (horizontal rule) between movie entries — this is the visual separator.
2. Use Markdown tables for single-movie detail views.
3. Use blockquotes (`>`) for overview / summary text.
4. Use bullet lists ONLY for cast members.
5. NEVER dump all fields as a flat paragraph.
6. If no context is relevant, say exactly: *"I don't have information about that in the database."*
7. Do NOT add any intro/outro fluff like "Here are the movies:" or "Hope this helps!" — go straight into the formatted content.

Context:
{context}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chroma_native_ef = embedding_functions.DefaultEmbeddingFunction()

    class LangChainChromaBuiltInEmbeddings:
        def embed_documents(self, texts):
            return chroma_native_ef(texts)

        def embed_query(self, text):
            return chroma_native_ef([text])[0]

    embeddings = LangChainChromaBuiltInEmbeddings()

    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    metadata_field_info = [
    AttributeInfo(
        name="title", 
        description="The title of the movie", 
        type="string"
    ),
    AttributeInfo(
        name="year", 
        description="The release year of the movie (e.g., 2023)", 
        type="integer"
    ),
    AttributeInfo(
        name="director", 
        description="The director's full name", 
        type="string"
    ),
    AttributeInfo(
        name="genres", 
        description="Comma-separated list of genres (e.g., 'Action, Drama, Sci-Fi')", 
        type="string"
    ),
    AttributeInfo(
        name="cast", 
        description="Comma-separated list of actor names", 
        type="string"
    ),
    AttributeInfo(
        name="rating", 
        description="Rating from 0 to 10", 
        type="float"
    ),
]

    retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    "Movie database with titles, years, directors, actors (cast), genres, and ratings. Use cast field for actor queries.",
    metadata_field_info,
    structured_query_translator=ChromaTranslator(),
    enable_limit=False,
    verbose=False,  # Set True for debugging
    search_kwargs={"k": 10}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


st.markdown(
    """
    <div class="ledger-masthead">
        <div class="kicker">Special Collections · Moving Image Division</div>
        <div class="title">Archive &amp; Ledger</div>
        <div class="subtitle">The Film Society Reading Room</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Retrieving the model & the card catalog from the stacks…"):
    rag_chain, retriever = load_chain()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand ──
    st.markdown(
        """
        <div style="text-align:center; padding:1.5rem 0 0.9rem 0; border-bottom:2px solid var(--ink); margin-bottom:0.5rem;">
            <div style="font-size:2rem; margin-bottom:0.3rem;">📇</div>
            <div style="font-family:var(--font-display); font-style:italic; font-size:1.3rem; font-weight:600; color:var(--ink); letter-spacing:-0.01em;">Archive &amp; Ledger</div>
            <div style="font-family:var(--font-mono); font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--rust); margin-top:0.3rem;">Film Society Reading Room</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Actions ──
    st.subheader("Ledger", anchor=False)
    if st.button("Clear the Reading Room", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # ── Tech stack table ──
    st.subheader("Catalog Specification", anchor=False)
    st.markdown(
        """
        <table style="width:100%; font-size:0.82rem; color:var(--ink-soft); border-collapse:collapse; font-family:var(--font-body);">
          <tr>
            <td style="padding:0.35rem 0; color:var(--ink-soft); border-bottom:1px solid var(--hairline);">LLM</td>
            <td style="text-align:right; border-bottom:1px solid var(--hairline);">
              <code style="background:var(--paper); border:1px solid var(--hairline); color:var(--rust); padding:0.1rem 0.4rem; border-radius:0; font-size:0.76rem; font-family:var(--font-mono);">llama-3.3-70b-versatile</code>
            </td>
          </tr>
          <tr>
            <td style="padding:0.35rem 0; color:var(--ink-soft); border-bottom:1px solid var(--hairline);">Provider</td>
            <td style="text-align:right; color:var(--ink); border-bottom:1px solid var(--hairline);">Groq</td>
          </tr>
          <tr>
            <td style="padding:0.35rem 0; color:var(--ink-soft); border-bottom:1px solid var(--hairline);">Embeddings</td>
            <td style="text-align:right; border-bottom:1px solid var(--hairline);">
              <code style="background:var(--paper); border:1px solid var(--hairline); color:var(--rust); padding:0.1rem 0.4rem; border-radius:0; font-size:0.76rem; font-family:var(--font-mono);">all-MiniLM-L6-v2</code>
            </td>
          </tr>
          <tr>
            <td style="padding:0.35rem 0; color:var(--ink-soft); border-bottom:1px solid var(--hairline);">Vector DB</td>
            <td style="text-align:right; color:var(--ink); border-bottom:1px solid var(--hairline);">Chroma (local)</td>
          </tr>
          <tr>
            <td style="padding:0.35rem 0; color:var(--ink-soft);">Retriever k</td>
            <td style="text-align:right; color:var(--ink);">10</td>
          </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Tips ──
    st.subheader("How to Consult the Catalog", anchor=False)
    st.markdown(
        """
        - Ask about a **specific film** for a detailed record card
        - Use *"suggest"* or *"recommend"* for a reading list
        - Filter by **genre**, **year**, or **director**
        - Every answer is drawn strictly from the archive's holdings
        """
    )

    st.markdown(
        '<div class="sidebar-footer">CATALOGED WITH STREAMLIT · LANGCHAIN · GROQ</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Helper: extract a movie title from a raw chunk for source labels ──
def _extract_title(chunk: str) -> str:
    for pattern in [r"Title:\s*(.+?)(?:\n|$)", r"^(.+?)(?:\n|$)"]:
        m = re.search(pattern, chunk)
        if m:
            title = m.group(1).strip()
            return title if len(title) <= 60 else title[:57] + "…"
    return "Untitled"


# ═══════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN  (only when chat is empty)
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="accession-tag" style="float:none; display:inline-block; margin-bottom:0.9rem;">EST. READING ROOM</div>
            <div class="welcome-icon">📇</div>
            <div class="welcome-title">Welcome to the Reading Room</div>
            <p class="welcome-sub">
                Consult the archive on any film in the collection.<br>
                Your inquiry will be searched against the holdings and returned as a catalog record.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Make suggestion chips actually clickable via Streamlit buttons
    suggestions = [
        "Suggest a drama movie",
        "Tell me about Inception",
        "Highest rated sci-fi films",
        "Movies by Christopher Nolan",
    ]
    cols = st.columns(2)
    for idx, text in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(text, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_question = text
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER PAST MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════
_accession_counter = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _accession_counter += 1
            st.markdown(
                f'<div class="accession-tag">NO. {_accession_counter:04d}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🗂️ Catalog Record — sources consulted", expanded=False):
                for i, (chunk_text, label) in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-chunk">'
                        f'<strong>Entry {i} — {label}</strong><br>'
                        f"{chunk_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ═══════════════════════════════════════════════════════════════════════════════
#  HANDLE INPUT  (from suggestion chip OR chat input box)
# ═══════════════════════════════════════════════════════════════════════════════

# Priority 1: pending question from a suggestion chip
user_question = st.session_state.pending_question
if user_question:
    st.session_state.pending_question = None  # consume it

# Priority 2: typed input from the chat box (only if no pending question)
if not user_question:
    user_question = st.chat_input(
        placeholder="e.g. What is the plot of Interstellar?",
    )

# ── Process the question ──────────────────────────────────────────────────────
if user_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):
        _new_accession = (
            sum(1 for m in st.session_state.messages if m["role"] == "assistant") + 1
        )
        st.markdown(
            f'<div class="accession-tag">NO. {_new_accession:04d}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Pulling the record & consulting the stacks…"):
            try:
                docs = retriever.invoke(user_question)
                sources = [
                    (
                        d.page_content[:250] + ("…" if len(d.page_content) > 250 else ""),
                        _extract_title(d.page_content),
                    )
                    for d in docs
                ]
                answer = rag_chain.invoke(user_question)
            except Exception as e:
                answer = f"⚠️ Something went wrong: `{e}`"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("🗂️ Catalog Record — sources consulted", expanded=False):
                for i, (chunk_text, label) in enumerate(sources, 1):
                    st.markdown(
                        f'<div class="source-chunk">'
                        f'<strong>Entry {i} — {label}</strong><br>'
                        f"{chunk_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )