"""
Inicialização centralizada do LLM (Groq) usado por todos os agentes.
Centralizar aqui evita configuração duplicada/divergente entre agentes.

Também concentra uma camada de recuperação para uma falha conhecida do
Groq/Llama ao usar tool calling com um "menu" de ações (campo "acao" com
enum): em vez de chamar a ferramenta de decisão combinada (ex.:
"decidir_acao_credito") com o campo "acao" preenchido, o modelo às vezes
"inventa" uma ferramenta separada com o nome da própria ação (ex.: tenta
chamar uma ferramenta "aceitar_entrevista", que não existe). Como essa
ferramenta não está na lista enviada à API, o Groq recusa a chamada
INTEIRA com HTTP 400 antes mesmo dela chegar até nós, então nem o
try/except dos agentes consegue recuperar a decisão — só mostrar erro.
`invocar_com_recuperacao` intercepta esse erro específico e remonta a
decisão a partir do payload que o próprio Groq devolve na mensagem de
erro, como se a ferramenta certa tivesse sido chamada desde o início.
"""
import json
import logging

from groq import BadRequestError
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("banco_agil.llm")


def get_llm(temperature: float = 0.2) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. Defina-a no arquivo .env "
            "(veja .env.example) ou nas variáveis de ambiente."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def invocar_com_recuperacao(llm, mensagens, nome_ferramenta_decisao: str) -> AIMessage:
    """
    Invoca o LLM normalmente. Se o Groq rejeitar a chamada porque o modelo
    tentou usar o nome de uma ação (ex.: "aceitar_entrevista") como se
    fosse uma ferramenta própria, tenta reconstruir a decisão a partir do
    payload retornado no erro, devolvendo uma AIMessage equivalente à que
    teríamos recebido se a ferramenta correta (`nome_ferramenta_decisao`)
    tivesse sido chamada. Qualquer outro tipo de erro é relançado
    normalmente, para ser tratado pelo try/except do agente chamador.
    """
    try:
        return llm.invoke(mensagens)
    except BadRequestError as exc:
        recuperada = _tentar_recuperar_tool_call(exc, nome_ferramenta_decisao)
        if recuperada is None:
            raise
        logger.warning(
            "Tool call inválida do Groq recuperada automaticamente: %s",
            recuperada.tool_calls,
        )
        return recuperada


def _tentar_recuperar_tool_call(exc: BadRequestError, nome_ferramenta_decisao: str):
    try:
        corpo = exc.body if isinstance(exc.body, dict) else {}
        erro = corpo.get("error", {})
        if erro.get("code") != "tool_use_failed":
            return None

        bruto = json.loads(erro.get("failed_generation") or "{}")
        nome_chamado = bruto.get("name")
        argumentos = bruto.get("arguments")
        if not isinstance(argumentos, dict):
            return None

        # O modelo às vezes já inclui "acao" correta nos argumentos, mesmo
        # tendo usado um nome de ferramenta inventado. Se não incluir,
        # usamos o próprio nome inventado como valor de "acao" (foi
        # exatamente o que ele quis dizer).
        argumentos.setdefault("acao", nome_chamado)

        return AIMessage(content="", tool_calls=[{
            "name": nome_ferramenta_decisao,
            "args": argumentos,
            "id": "recuperado_0",
            "type": "tool_call",
        }])
    except Exception:
        return None
