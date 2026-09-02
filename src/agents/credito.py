"""
Nó do Agente de Crédito no grafo LangGraph.

Consulta/atualiza limite de crédito e, em caso de rejeição do pedido de
aumento, oferece (e encaminha implicitamente para) a Entrevista de Crédito.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm import get_llm, invocar_com_recuperacao
from src.prompts import montar_prompt
from src.state import BancoAgilState
from src.tools.csv_tools import registrar_solicitacao_aumento_limite

logger = logging.getLogger("banco_agil.agents.credito")

# Respostas claramente afirmativas/negativas, tratadas em código (sem LLM)
# quando há uma solicitação de aumento aguardando confirmação. Isso evita
# depender de o modelo classificar corretamente um "sim"/"não" simples —
# que é uma decisão crítica, já que registra uma mudança real no limite
# do cliente. Respostas fora dessas listas continuam indo para o LLM
# normalmente (ex.: "sim, mas queria um valor diferente").
_RESPOSTAS_AFIRMATIVAS = {
    "sim", "s", "yes", "confirmo", "confirmado", "pode seguir", "pode ser",
    "isso", "isso mesmo", "certo", "afirmativo", "ok", "okay", "beleza",
    "positivo", "concordo", "quero", "claro", "com certeza",
}
_RESPOSTAS_NEGATIVAS = {
    "não", "nao", "n", "no", "cancela", "cancelar", "cancele", "desisto",
    "negativo", "não quero", "nao quero", "deixa pra lá", "deixa pra la",
    "esquece", "melhor não", "melhor nao",
}


def _ultima_mensagem_humana(state: BancoAgilState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _interpretar_confirmacao(texto):
    if not texto:
        return None
    limpo = texto.strip().lower().strip(".! ")
    if limpo in _RESPOSTAS_AFIRMATIVAS:
        return "confirmar"
    if limpo in _RESPOSTAS_NEGATIVAS:
        return "cancelar"
    return None

DECISAO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "decidir_acao_credito",
        "description": (
            "Registra a decisão do agente de crédito para a mensagem atual do cliente. "
            "Chame SEMPRE, em toda mensagem, com a ação mais adequada."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "acao": {
                    "type": "string",
                    "enum": [
                        "consultar_limite",
                        "consultar_score",
                        "solicitar_aumento",
                        "confirmar_solicitacao",
                        "cancelar_solicitacao",
                        "aceitar_entrevista",
                        "recusar_entrevista",
                        "mudar_assunto",
                        "encerrar",
                        "conversar",
                    ],
                    "description": (
                        "consultar_limite: cliente quer saber o limite de crédito atual. "
                        "consultar_score: cliente quer saber seu score de crédito atual. "
                        "solicitar_aumento: use APENAS quando o cliente já informou um "
                        "VALOR NUMÉRICO claro para o novo limite desejado (nesse caso "
                        "preencha novo_limite_solicitado com esse número) — isso apenas "
                        "APRESENTA o pedido para confirmação, NÃO registra ainda. Se o "
                        "cliente disse apenas que quer aumentar o limite mas AINDA NÃO deu "
                        "um valor, use 'conversar' e pergunte o valor desejado em "
                        "resposta_ao_cliente — NUNCA chame solicitar_aumento sem um número. "
                        "confirmar_solicitacao: cliente confirmou (sim/confirmo/pode seguir) "
                        "uma solicitação de aumento que você apresentou e está aguardando "
                        "confirmação. "
                        "cancelar_solicitacao: cliente NÃO confirmou / desistiu da "
                        "solicitação de aumento que estava aguardando confirmação. "
                        "aceitar_entrevista: cliente aceitou fazer a entrevista de crédito "
                        "oferecida após uma rejeição. "
                        "recusar_entrevista: cliente recusou a entrevista oferecida. "
                        "mudar_assunto: cliente quer tratar de outro assunto que NÃO é "
                        "crédito (ex.: cotação de moedas) — preencha novo_assunto. "
                        "encerrar: cliente pediu para encerrar o atendimento. "
                        "conversar: qualquer outra resposta (ex.: ainda faltam dados, "
                        "cliente fazendo pergunta geral sobre crédito) — nesse caso "
                        "responda normalmente."
                    ),
                },
                "novo_limite_solicitado": {
                    "type": ["number", "null"],
                    "description": (
                        "Valor do novo limite desejado, obrigatório quando "
                        "acao=solicitar_aumento. Deixe null/omita em qualquer outra ação, "
                        "inclusive quando o cliente ainda não informou o valor."
                    ),
                },
                "novo_assunto": {
                    "type": ["string", "null"],
                    "enum": ["cambio", None],
                    "description": "Obrigatório quando acao=mudar_assunto: para qual assunto o cliente quer ir.",
                },
                "resposta_ao_cliente": {
                    "type": ["string", "null"],
                    "description": (
                        "Texto a ser dito ao cliente AGORA, usado apenas quando acao=conversar "
                        "(ex.: pedir o valor do novo limite desejado)."
                    ),
                },
            },
            "required": ["acao"],
        },
    },
}


def node_credito(state: BancoAgilState) -> dict:
    try:
        return _executar(state)
    except Exception as exc:
        logger.exception("Erro inesperado no Agente de Crédito: %s", exc)
        return {
            "messages": [AIMessage(content=(
                "Tivemos uma instabilidade técnica ao tratar sua solicitação de crédito. "
                "Pode tentar novamente em instantes?"
            ))],
            "log_erros": state.get("log_erros", []) + [f"credito: {exc}"],
            "continuar_mesmo_turno": False,
        }


def _executar(state: BancoAgilState) -> dict:
    cpf = state.get("cpf_cliente")
    nome = state.get("nome_cliente", "cliente")
    limite_atual = state.get("limite_credito_atual")
    score = state.get("score_atual")

    # Retorno automático da Entrevista de Crédito: reavalia o pedido
    # pendente diretamente (sem depender do LLM interpretar a última
    # mensagem, que pertence à entrevista, não a uma nova instrução).
    if state.get("solicitar_reavaliacao_credito") and state.get("novo_limite_desejado") is not None:
        return _reavaliar_apos_entrevista(state, cpf, limite_atual)

    pendente = state.get("novo_limite_pendente_confirmacao")
    if pendente is not None:
        decisao = _interpretar_confirmacao(_ultima_mensagem_humana(state))
        if decisao == "confirmar":
            return _registrar_aumento(cpf, limite_atual, pendente)
        if decisao == "cancelar":
            texto = "Sem problemas, cancelei essa solicitação. Posso ajudar em mais alguma coisa?"
            return {
                "messages": [AIMessage(content=texto)],
                "novo_limite_pendente_confirmacao": None,
                "agente_atual": "triagem",
                "continuar_mesmo_turno": False,
            }
        # Resposta ambígua (ex.: "sim, mas queria outro valor"): segue para
        # o LLM normalmente, que já tem instrução para tratar esse caso.

    contexto = (
        f"\n\nContexto do cliente autenticado (não pergunte esses dados novamente): "
        f"nome={nome}, limite_credito_atual=R$ {limite_atual:.2f}, score_atual={score}."
    )
    system = SystemMessage(content=montar_prompt("credito") + contexto)
    llm = get_llm().bind_tools([DECISAO_SCHEMA])
    resposta = invocar_com_recuperacao(llm, [system] + state["messages"], "decidir_acao_credito")

    tool_calls = getattr(resposta, "tool_calls", None) or []
    decisao_call = next((c for c in tool_calls if c["name"] == "decidir_acao_credito"), None)

    if not decisao_call:
        texto = resposta.content or "Como posso ajudar com seu crédito hoje?"
        return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

    args = decisao_call["args"]
    acao = args.get("acao", "conversar")

    if acao == "encerrar":
        return {
            "messages": [AIMessage(content="Tudo bem! Obrigado por falar com o Banco Ágil. Até breve!")],
            "encerrar_conversa": True,
            "agente_atual": "encerrado",
        }

    if acao == "consultar_limite":
        texto = f"Seu limite de crédito disponível atualmente é de R$ {limite_atual:.2f}. Deseja solicitar um aumento?"
        return {"messages": [AIMessage(content=texto)], "agente_atual": "triagem", "continuar_mesmo_turno": False}

    if acao == "consultar_score":
        texto = f"Seu score de crédito atual é {score:.0f}. Posso ajudar em mais alguma coisa?"
        return {"messages": [AIMessage(content=texto)], "agente_atual": "triagem", "continuar_mesmo_turno": False}

    if acao == "mudar_assunto":
        novo_assunto = args.get("novo_assunto")
        if novo_assunto == "cambio":
            return {
                "agente_atual": "cambio",
                "messages": [],
                "continuar_mesmo_turno": True,
                "cambio_moeda_perguntada": False,
            }
        texto = "Você deseja continuar falando sobre seu crédito, ou consultar a cotação de alguma moeda?"
        return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

    if acao == "solicitar_aumento":
        novo_limite = args.get("novo_limite_solicitado")
        if novo_limite is None:
            texto = "Qual seria o novo limite de crédito desejado?"
            return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

        try:
            novo_limite = float(novo_limite)
        except (TypeError, ValueError):
            texto = "Não consegui entender esse valor. Pode informar o novo limite desejado novamente?"
            return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

        texto = (
            f"Só para confirmar: você quer solicitar o aumento do seu limite de "
            f"R$ {limite_atual:.2f} para R$ {novo_limite:.2f}. Posso seguir com essa "
            "solicitação?"
        )
        return {
            "messages": [AIMessage(content=texto)],
            "novo_limite_pendente_confirmacao": novo_limite,
            "continuar_mesmo_turno": False,
        }

    if acao == "confirmar_solicitacao":
        pendente = state.get("novo_limite_pendente_confirmacao")
        if pendente is None:
            texto = "Não há nenhuma solicitação de aumento aguardando confirmação no momento. Deseja solicitar um novo limite?"
            return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}
        return _registrar_aumento(cpf, limite_atual, pendente)

    if acao == "cancelar_solicitacao":
        texto = "Sem problemas, cancelei essa solicitação. Posso ajudar em mais alguma coisa?"
        return {
            "messages": [AIMessage(content=texto)],
            "novo_limite_pendente_confirmacao": None,
            "agente_atual": "triagem",
            "continuar_mesmo_turno": False,
        }

    if acao == "aceitar_entrevista":
        return {
            "agente_atual": "entrevista",
            "messages": [],
            "continuar_mesmo_turno": True,
            "entrevista_em_andamento": True,
            "entrevista_respostas": {},
        }

    if acao == "recusar_entrevista":
        texto = "Sem problemas! Posso ajudar em mais alguma coisa, ou deseja encerrar o atendimento?"
        return {"messages": [AIMessage(content=texto)], "agente_atual": "triagem", "continuar_mesmo_turno": False}

    # acao == "conversar"
    texto = args.get("resposta_ao_cliente") or resposta.content or "Como posso ajudar com seu crédito hoje?"
    return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}


def _registrar_aumento(cpf: str, limite_atual: float, novo_limite: float) -> dict:
    """Efetivamente registra a solicitação de aumento de limite — só é
    chamada depois que o cliente CONFIRMA o pedido explicitamente."""
    resultado = registrar_solicitacao_aumento_limite.invoke({
        "cpf": cpf,
        "limite_atual": limite_atual,
        "novo_limite_solicitado": novo_limite,
    })

    base_update = {"novo_limite_pendente_confirmacao": None, "continuar_mesmo_turno": False}

    if not resultado.get("sucesso"):
        texto = resultado.get("erro", "Não foi possível registrar sua solicitação agora. Pode tentar novamente?")
        return {**base_update, "messages": [AIMessage(content=texto)]}

    status = resultado["status_pedido"]

    if status == "aprovado":
        texto = (
            f"Boas notícias! Sua solicitação de aumento de limite para "
            f"R$ {float(novo_limite):.2f} foi aprovada. Seu novo limite já está ativo. "
            "Posso ajudar em mais alguma coisa?"
        )
        return {
            **base_update,
            "messages": [AIMessage(content=texto)],
            "limite_credito_atual": float(novo_limite),
            "ultima_solicitacao_status": "aprovado",
            "agente_atual": "triagem",
        }

    texto = (
        f"Infelizmente, sua solicitação de aumento de limite para R$ {float(novo_limite):.2f} "
        "não foi aprovada com base no seu score de crédito atual. Gostaria de fazer uma "
        "entrevista de crédito rápida para tentar atualizar seu score e reavaliar o pedido?"
    )
    return {
        **base_update,
        "messages": [AIMessage(content=texto)],
        "ultima_solicitacao_status": "rejeitado",
        "novo_limite_desejado": float(novo_limite),
    }


def _reavaliar_apos_entrevista(state: BancoAgilState, cpf: str, limite_atual: float) -> dict:
    novo_limite = state["novo_limite_desejado"]

    resultado = registrar_solicitacao_aumento_limite.invoke({
        "cpf": cpf,
        "limite_atual": limite_atual,
        "novo_limite_solicitado": novo_limite,
    })

    base_update = {"solicitar_reavaliacao_credito": False, "novo_limite_desejado": None, "continuar_mesmo_turno": False}

    if not resultado.get("sucesso"):
        texto = resultado.get("erro", "Não foi possível reavaliar sua solicitação agora. Pode tentar novamente?")
        return {**base_update, "messages": [AIMessage(content=texto)]}

    status = resultado["status_pedido"]

    if status == "aprovado":
        texto = (
            f"Ótima notícia! Com seu score atualizado, sua solicitação de aumento de limite "
            f"para R$ {float(novo_limite):.2f} foi aprovada. Posso ajudar em mais alguma coisa?"
        )
        return {
            **base_update,
            "messages": [AIMessage(content=texto)],
            "limite_credito_atual": float(novo_limite),
            "ultima_solicitacao_status": "aprovado",
            "agente_atual": "triagem",
        }

    texto = (
        f"Mesmo com o score atualizado, a solicitação de aumento para R$ {float(novo_limite):.2f} "
        "ainda não pôde ser aprovada no momento. Posso ajudar em mais alguma coisa?"
    )
    return {
        **base_update,
        "messages": [AIMessage(content=texto)],
        "ultima_solicitacao_status": "rejeitado",
        "agente_atual": "triagem",
    }
