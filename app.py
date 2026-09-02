"""
Interface Streamlit do Banco Ágil.

Recursos de UX implementados (além do fluxo funcional dos agentes):
1. O campo de mensagem fica DESABILITADO enquanto o agente está
   processando a resposta anterior — o cliente só pode enviar uma nova
   mensagem depois que o agente respondeu.
2. Design moderno, com cartões, avatares e indicador de "digitando...",
   adaptado automaticamente a tema claro OU escuro (usa as variáveis de
   CSS nativas do Streamlit, que mudam conforme o tema ativo escolhido
   pelo usuário no menu ⋮ > Settings > Theme).
"""
import sys
import time
import html
import re
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import GROQ_API_KEY
from src.graph import get_graph

st.set_page_config(
    page_title="Banco Ágil — Atendimento Inteligente",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="auto",
)

# --------------------------------------------------------------------------
# Tema (claro/escuro) controlado por um botão nosso — independente do menu
# de configurações do Streamlit.
# --------------------------------------------------------------------------
if "tema" not in st.session_state:
    st.session_state.tema = "light"

_CORES_TEMA = {
    "light": {
        "primary": "#6C5CE7",
        "bg": "#ffffff",
        "bg_secondary": "#f5f3ff",
        "text": "#1e1b2e",
    },
    "dark": {
        "primary": "#a29bfe",
        "bg": "#111018",
        "bg_secondary": "#1c1a29",
        "text": "#f2f0ff",
    },
}
_cores = _CORES_TEMA[st.session_state.tema]


def _formatar_mensagem(texto: str) -> str:
    """Converte markdown básico (**negrito**, quebras de linha) para HTML,
    escapando o restante — necessário porque o parser de markdown do
    Streamlit não processa markdown dentro de uma <div> customizada."""
    escapado = html.escape(texto or "")
    escapado = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escapado)
    escapado = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escapado)
    escapado = escapado.replace("\n", "<br>")
    return escapado

# --------------------------------------------------------------------------
# CSS — visual moderno, com suporte a tema claro/escuro controlado pelo
# botão da barra lateral (variáveis --ba-* recalculadas conforme o tema).
# --------------------------------------------------------------------------
_root_vars = f"""
    :root {{
        --ba-primary: {_cores['primary']};
        --ba-bg: {_cores['bg']};
        --ba-bg-secondary: {_cores['bg_secondary']};
        --ba-text: {_cores['text']};
    }}
    .stApp {{
        background-color: var(--ba-bg);
        color: var(--ba-text);
    }}
"""

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    html, body {
        background-color: var(--ba-bg) !important;
    }

    [data-testid="stHeader"] {
        background-color: var(--ba-bg) !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--ba-bg-secondary) !important;
        border-right: 1px solid rgba(108, 92, 231, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: var(--ba-text) !important;
    }
    [data-testid="stSidebar"] button {
        background-color: var(--ba-bg) !important;
        color: var(--ba-text) !important;
        border: 1px solid rgba(108, 92, 231, 0.35) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(108, 92, 231, 0.2) !important;
    }
    [data-testid="stExpander"] {
        background-color: var(--ba-bg-secondary) !important;
        border: 1px solid rgba(108, 92, 231, 0.25) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        background-color: var(--ba-bg-secondary) !important;
        color: var(--ba-text) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--ba-primary) !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: var(--ba-bg-secondary) !important;
        color: var(--ba-text) !important;
    }

    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div {
        background-color: var(--ba-bg) !important;
    }
    [data-testid="stChatInput"] {
        background-color: var(--ba-bg-secondary) !important;
        border: 1px solid rgba(108, 92, 231, 0.3) !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] div {
        background-color: var(--ba-bg-secondary) !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: var(--ba-text) !important;
        -webkit-text-fill-color: var(--ba-text) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--ba-text) !important;
        opacity: 0.55 !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background-color: var(--ba-primary) !important;
    }

    """
    + _root_vars
    + """

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 760px;
    }

    .ba-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 22px;
        border-radius: 18px;
        background: linear-gradient(135deg, var(--ba-primary) 0%, #a29bfe 100%);
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.25);
        margin-bottom: 22px;
    }
    .ba-header .ba-logo {
        font-size: 34px;
        line-height: 1;
    }
    .ba-header h1 {
        color: #ffffff;
        font-size: 21px;
        font-weight: 700;
        margin: 0;
    }
    .ba-header p {
        color: rgba(255,255,255,0.88);
        font-size: 13px;
        margin: 2px 0 0 0;
    }

    .ba-bubble {
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .ba-bubble-assistant {
        background: var(--ba-bg-secondary);
        color: var(--ba-text);
        border: 1px solid rgba(108, 92, 231, 0.12);
        border-top-left-radius: 4px;
    }
    .ba-bubble-user {
        background: linear-gradient(135deg, var(--ba-primary) 0%, #8378f0 100%);
        color: #ffffff;
        border-top-right-radius: 4px;
    }

    .ba-typing {
        display: inline-flex;
        gap: 4px;
        padding: 14px 16px;
        border-radius: 16px;
        background: var(--ba-bg-secondary);
        border: 1px solid rgba(108, 92, 231, 0.12);
    }
    .ba-typing span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--ba-primary);
        opacity: 0.5;
        animation: ba-bounce 1.1s infinite ease-in-out;
    }
    .ba-typing span:nth-child(2) { animation-delay: 0.15s; }
    .ba-typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes ba-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-5px); opacity: 1; }
    }

    .ba-status-card {
        border-radius: 14px;
        padding: 14px 16px;
        background: var(--ba-bg-secondary);
        border: 1px solid rgba(108, 92, 231, 0.15);
        margin-bottom: 10px;
    }
    .ba-status-card b { color: var(--ba-primary); }
    .ba-status-row {
        display: flex;
        justify-content: space-between;
        font-size: 13.5px;
        padding: 3px 0;
        color: var(--ba-text);
    }

    .ba-encerrado {
        text-align: center;
        padding: 18px;
        border-radius: 14px;
        background: var(--ba-bg-secondary);
        color: var(--ba-text);
        font-size: 14px;
        border: 1px dashed rgba(108, 92, 231, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Estado da sessão
# --------------------------------------------------------------------------
GREETING = (
    "Olá! 👋 Bem-vindo(a) ao **Banco Ágil**. Para começarmos, preciso confirmar "
    "sua identidade — pode me informar seu **CPF** e sua **data de nascimento**?"
)

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [{"role": "assistant", "content": GREETING}]

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [AIMessage(content=GREETING)],
        "agente_atual": "triagem",
        "autenticado": False,
        "tentativas_autenticacao": 0,
        "encerrar_conversa": False,
        "continuar_mesmo_turno": False,
        "log_erros": [],
    }

if "processing" not in st.session_state:
    st.session_state.processing = False

if "encerrado" not in st.session_state:
    st.session_state.encerrado = False


def _reiniciar_atendimento():
    st.session_state.ui_messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.graph_state = {
        "messages": [AIMessage(content=GREETING)],
        "agente_atual": "triagem",
        "autenticado": False,
        "tentativas_autenticacao": 0,
        "encerrar_conversa": False,
        "continuar_mesmo_turno": False,
        "log_erros": [],
    }
    st.session_state.processing = False
    st.session_state.encerrado = False


# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="ba-header">
        <div class="ba-logo">🏦</div>
        <div>
            <h1>Banco Ágil</h1>
            <p>Atendimento inteligente • Crédito • Câmbio</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Barra lateral: status da sessão
# --------------------------------------------------------------------------
with st.sidebar:
    rotulo_tema = "🌙 Modo escuro" if st.session_state.tema == "light" else "☀️ Modo claro"
    if st.button(rotulo_tema, use_container_width=True):
        st.session_state.tema = "dark" if st.session_state.tema == "light" else "light"
        st.rerun()

    st.markdown("### 🧾 Sua sessão")
    gs = st.session_state.graph_state
    if gs.get("autenticado"):
        nome = gs.get("nome_cliente", "—")
        limite = gs.get("limite_credito_atual")
        score = gs.get("score_atual")
        st.markdown(
            f"""
            <div class="ba-status-card">
                <div class="ba-status-row"><span>Cliente</span><b>{nome}</b></div>
                <div class="ba-status-row"><span>Limite atual</span><b>R$ {limite:,.2f}</b></div>
                <div class="ba-status-row"><span>Score</span><b>{score:.0f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Você ainda não foi autenticado nesta conversa.")

    st.divider()
    if st.button("🔄 Novo atendimento", use_container_width=True):
        _reiniciar_atendimento()
        st.rerun()

    with st.expander("🧪 Dados de teste"):
        st.caption("Use qualquer um destes CPFs (em qualquer formato) para autenticar:")
        st.markdown(
            """
            <div class="ba-status-card" style="margin-bottom:0;">
                <div class="ba-status-row"><span>Ana Beatriz</span><span></span></div>
                <div class="ba-status-row"><span>CPF</span><b>104.332.181-00</b></div>
                <div class="ba-status-row"><span>Nasc.</span><b>12/04/1990</b></div>
                <div class="ba-status-row"><span>Score</span><b>620</b></div>
                <hr style="margin:8px 0; border-color: rgba(108,92,231,0.2);">
                <div class="ba-status-row"><span>João Pedro</span><span></span></div>
                <div class="ba-status-row"><span>CPF</span><b>026.542.351-14</b></div>
                <div class="ba-status-row"><span>Nasc.</span><b>20/11/1975</b></div>
                <div class="ba-status-row"><span>Score</span><b>180</b></div>
                <hr style="margin:8px 0; border-color: rgba(108,92,231,0.2);">
                <div class="ba-status-row"><span>Fernanda Costa</span><span></span></div>
                <div class="ba-status-row"><span>CPF</span><b>083.863.794-99</b></div>
                <div class="ba-status-row"><span>Nasc.</span><b>05/01/1998</b></div>
                <div class="ba-status-row"><span>Score</span><b>740</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("")

    if not GROQ_API_KEY:
        st.warning(
            "GROQ_API_KEY não configurada. Defina-a no arquivo .env "
            "para o assistente funcionar.",
            icon="⚠️",
        )

# --------------------------------------------------------------------------
# Histórico da conversa
# --------------------------------------------------------------------------
for msg in st.session_state.ui_messages:
    avatar = "🏦" if msg["role"] == "assistant" else "🙂"
    bubble_class = "ba-bubble-assistant" if msg["role"] == "assistant" else "ba-bubble-user"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(
            f'<div class="ba-bubble {bubble_class}">{_formatar_mensagem(msg["content"])}</div>',
            unsafe_allow_html=True,
        )

if st.session_state.encerrado:
    st.markdown(
        '<div class="ba-encerrado">🔒 Este atendimento foi encerrado. '
        'Clique em <b>"Novo atendimento"</b> na barra lateral para começar uma nova conversa.</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Processamento da resposta do agente | input só libera depois
# que o agente responder 
# --------------------------------------------------------------------------
if st.session_state.processing:
    with st.chat_message("assistant", avatar="🏦"):
        st.markdown(
            '<div class="ba-typing"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )

    try:
        graph = get_graph()
        estado_antes = st.session_state.graph_state
        qtd_antes = len(estado_antes["messages"])

        estado_antes["continuar_mesmo_turno"] = False
        resultado = graph.invoke(estado_antes, config={"recursion_limit": 20})

        st.session_state.graph_state = resultado

        novas_mensagens = resultado["messages"][qtd_antes:]
        for m in novas_mensagens:
            if isinstance(m, AIMessage) and m.content:
                st.session_state.ui_messages.append({"role": "assistant", "content": m.content})

        if resultado.get("encerrar_conversa"):
            st.session_state.encerrado = True

    except RuntimeError as exc:
        st.session_state.ui_messages.append({
            "role": "assistant",
            "content": f"⚠️ {exc}",
        })
    except Exception as exc:  # pragma: no cover - proteção geral de UI
        st.session_state.ui_messages.append({
            "role": "assistant",
            "content": (
                "⚠️ Tivemos uma instabilidade técnica inesperada. Pode tentar "
                "novamente em instantes?"
            ),
        })

    st.session_state.processing = False
    st.rerun()

# --------------------------------------------------------------------------
# Campo de entrada desabilitado enquanto o agente está processando ou o
# atendimento foi encerrado.
# --------------------------------------------------------------------------
input_desabilitado = st.session_state.processing or st.session_state.encerrado
placeholder = (
    "Aguarde a resposta do atendente..." if st.session_state.processing
    else "Atendimento encerrado" if st.session_state.encerrado
    else "Digite sua mensagem..."
)

prompt = st.chat_input(placeholder, disabled=input_desabilitado)

# Trava de segurança no lado do servidor: o campo já vem desabilitado
# visualmente (linha acima), mas isso sozinho não impede 100% dos casos
# (ex.: alguma inconsistência de re-render). Aqui garantimos que nenhuma
# mensagem é processada enquanto o atendimento estiver encerrado ou uma
# resposta já estiver em andamento, mesmo que o campo tenha sido burlado.
if prompt and not st.session_state.encerrado and not st.session_state.processing:
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    st.session_state.graph_state["messages"].append(HumanMessage(content=prompt))
    st.session_state.processing = True
    st.rerun()