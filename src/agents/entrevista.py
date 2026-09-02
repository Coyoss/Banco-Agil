"""
Nó do Agente de Entrevista de Crédito no grafo LangGraph.

Coleta, uma pergunta por vez, os cinco dados financeiros exigidos pelo
desafio, calcula o novo score (via ferramenta determinística) e devolve
o cliente ao Agente de Crédito para nova análise — tudo de forma
implícita, sem o cliente perceber a "troca" de especialista.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.triagem import MENU_ASSUNTOS
from src.state import BancoAgilState
from src.tools.csv_tools import atualizar_score_cliente
from src.tools.score_tools import calcular_novo_score
from src.validators import parse_numero

logger = logging.getLogger("banco_agil.agents.entrevista")

# Ordem fixa das perguntas da entrevista.
CAMPOS = [
    ("renda_mensal", "Para começar, qual é a sua renda mensal aproximada?"),
    ("tipo_emprego", "Qual o seu tipo de emprego atual: formal (CLT), autônomo ou desempregado?"),
    ("despesas_fixas_mensais", "Qual o valor aproximado das suas despesas fixas mensais?"),
    ("numero_dependentes", "Quantos dependentes você possui (0, 1, 2, 3 ou mais)?"),
    ("possui_dividas_ativas", "Você possui dívidas ativas no momento? (sim/não)"),
]

# Palavras/expressões que indicam que o cliente quer abandonar a
# entrevista a qualquer momento. Checagem por substring (não exata) já
# que aqui a variação de frase costuma ser maior do que num simples
# sim/não — e nenhum campo da entrevista (número, tipo de emprego,
# dependentes, sim/não de dívidas) tende a conter essas palavras.
_PALAVRAS_CANCELAMENTO = (
    "sair", "cancela", "desist", "parar", "encerrar",
    "não quero mais", "nao quero mais",
    "não quero continuar", "nao quero continuar",
    "voltar", "deixa pra lá", "deixa pra la",
)


def _cliente_quer_sair(texto: str) -> bool:
    if not texto:
        return False
    limpo = texto.strip().lower()
    return any(palavra in limpo for palavra in _PALAVRAS_CANCELAMENTO)


def node_entrevista(state: BancoAgilState) -> dict:
    try:
        return _executar(state)
    except Exception as exc:
        logger.exception("Erro inesperado no Agente de Entrevista: %s", exc)
        return {
            "messages": [AIMessage(content=(
                "Tivemos uma instabilidade técnica durante a entrevista de crédito. "
                "Pode tentar responder novamente?"
            ))],
            "log_erros": state.get("log_erros", []) + [f"entrevista: {exc}"],
            "continuar_mesmo_turno": False,
        }


def _ultima_mensagem_humana(state: BancoAgilState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _executar(state: BancoAgilState) -> dict:
    respostas = dict(state.get("entrevista_respostas") or {})
    idx = len(respostas)

    # Primeira entrada na entrevista (handoff vindo do Agente de Crédito):
    # ainda não há nenhuma resposta registrada e a última mensagem do
    # histórico não é uma resposta a pergunta nenhuma -> apenas pergunta.
    ja_iniciada = state.get("entrevista_em_andamento") and idx > 0

    if idx < len(CAMPOS) and not _ultima_resposta_pertence_a_pergunta_pendente(state, idx):
        # Ainda não coletamos essa resposta: pergunta atual permanece pendente.
        pergunta = CAMPOS[idx][1]
        return {
            "messages": [AIMessage(content=pergunta)],
            "entrevista_em_andamento": True,
            "entrevista_respostas": respostas,
            "continuar_mesmo_turno": False,
        }

    # Processa a resposta do usuário para a pergunta pendente.
    texto_usuario = _ultima_mensagem_humana(state) or ""

    if _cliente_quer_sair(texto_usuario):
        return _sair_da_entrevista()

    campo_atual, _ = CAMPOS[idx]
    valor, erro = _validar_campo(campo_atual, texto_usuario)
    if erro:
        return {
            "messages": [AIMessage(content=f"{erro} {CAMPOS[idx][1]}")],
            "entrevista_em_andamento": True,
            "entrevista_respostas": respostas,
            "continuar_mesmo_turno": False,
        }

    respostas[campo_atual] = valor

    if len(respostas) < len(CAMPOS):
        proxima_pergunta = CAMPOS[len(respostas)][1]
        return {
            "messages": [AIMessage(content=proxima_pergunta)],
            "entrevista_em_andamento": True,
            "entrevista_respostas": respostas,
            "continuar_mesmo_turno": False,
        }

    # Todas as respostas coletadas: calcula e grava o novo score.
    resultado_score = calcular_novo_score.invoke(respostas)
    if not resultado_score.get("sucesso"):
        return {
            "messages": [AIMessage(content=(
                "Não consegui calcular seu novo score com os dados informados. "
                "Podemos tentar novamente mais tarde. Deseja fazer mais alguma coisa?"
            ))],
            "entrevista_em_andamento": False,
            "entrevista_respostas": {},
            "agente_atual": "credito",
            "continuar_mesmo_turno": False,
        }

    novo_score = resultado_score["score"]
    atualizacao = atualizar_score_cliente.invoke({"cpf": state.get("cpf_cliente"), "novo_score": novo_score})

    if not atualizacao.get("sucesso"):
        texto = atualizacao.get("erro", "Não foi possível salvar seu novo score agora.")
        return {
            "messages": [AIMessage(content=texto)],
            "entrevista_em_andamento": False,
            "entrevista_respostas": {},
            "agente_atual": "credito",
            "continuar_mesmo_turno": False,
        }

    mensagem_transicao = (
        f"Ótimo, {state.get('nome_cliente', 'cliente').split()[0]}! Seu score foi atualizado "
        f"para {novo_score:.0f}. Vou verificar novamente sua solicitação de aumento de limite."
    )
    return {
        "messages": [AIMessage(content=mensagem_transicao)],
        "entrevista_em_andamento": False,
        "entrevista_respostas": {},
        "score_atual": novo_score,
        "agente_atual": "credito",
        "continuar_mesmo_turno": True,
        "solicitar_reavaliacao_credito": True,
    }


def _sair_da_entrevista() -> dict:
    texto = (
        "Sem problemas, vamos parar por aqui a entrevista de crédito — "
        "nenhuma informação foi salva.\n\n"
        f"{MENU_ASSUNTOS}"
    )
    return {
        "messages": [AIMessage(content=texto)],
        "entrevista_em_andamento": False,
        "entrevista_respostas": {},
        "solicitar_reavaliacao_credito": False,
        "novo_limite_desejado": None,
        "agente_atual": "triagem",
        "continuar_mesmo_turno": False,
    }


def _ultima_resposta_pertence_a_pergunta_pendente(state: BancoAgilState, idx: int) -> bool:
    """
    Verdadeiro quando a última mensagem do histórico é do humano E veio
    depois da pergunta atual ter sido feita (ou seja, é uma resposta a
    processar). Falso logo no handoff inicial (ainda não perguntamos nada).
    """
    if idx > 0:
        return True  # já estamos no meio da entrevista, sempre há pergunta pendente
    # idx == 0: só tratamos como resposta pendente se já existir pelo menos
    # uma pergunta do assistente sobre a entrevista no histórico recente.
    for msg in reversed(state["messages"][:-1] if state["messages"] else []):
        if isinstance(msg, AIMessage) and msg.content.endswith(CAMPOS[0][1]):
            return True
        if isinstance(msg, HumanMessage):
            continue
        break
    return False


def _validar_campo(campo: str, texto: str):
    if campo in ("renda_mensal", "despesas_fixas_mensais"):
        return parse_numero(texto)
    if campo == "tipo_emprego":
        texto_l = texto.strip().lower()
        if not texto_l:
            return None, "Não entendi seu tipo de emprego."
        return texto_l, None
    if campo == "numero_dependentes":
        texto_l = texto.strip().lower()
        if not texto_l:
            return None, "Não entendi o número de dependentes."
        return texto_l, None
    if campo == "possui_dividas_ativas":
        texto_l = texto.strip().lower()
        if not texto_l:
            return None, "Não entendi. Você possui dívidas ativas? (sim/não)"
        return texto_l, None
    return None, "Campo desconhecido."
