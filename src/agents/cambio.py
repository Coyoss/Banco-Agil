"""
Nó do Agente de Câmbio no grafo LangGraph.
"""
import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.llm import get_llm, invocar_com_recuperacao
from src.prompts import montar_prompt
from src.state import BancoAgilState
from src.tools.cambio_tools import consultar_cotacao

logger = logging.getLogger("banco_agil.agents.cambio")

# Menu apresentado quando o cliente ainda não especificou qual moeda quer
# consultar — evita que o agente assuma dólar (USD) por padrão sem
# perguntar. Lista curada de moedas comuns; o cliente pode digitar
# qualquer outro código ISO diretamente, mesmo fora desta lista.
MENU_MOEDAS = (
    "Posso consultar a cotação das principais moedas em relação ao Real (BRL). "
    "Qual você deseja?\n"
    "1. Dólar americano (USD)\n"
    "2. Euro (EUR)\n"
    "3. Libra esterlina (GBP)\n"
    "4. Peso argentino (ARS)\n"
    "5. Iene japonês (JPY)\n"
    "Se preferir outra moeda, pode informar o nome ou o código (ex.: CAD, CHF, CNY)."
)

DECISAO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "decidir_acao_cambio",
        "description": "Registra qual moeda o cliente quer consultar, se quer mudar de assunto, ou se pediu para encerrar.",
        "parameters": {
            "type": "object",
            "properties": {
                "acao": {
                    "type": "string",
                    "enum": ["consultar_cotacao", "mudar_assunto", "encerrar", "conversar"],
                    "description": (
                        "consultar_cotacao: cliente quer saber a cotação de uma moeda — "
                        "use esta ação MESMO QUE ele ainda não tenha dito qual moeda "
                        "(nesse caso deixe moeda_origem em branco/null; o sistema "
                        "pergunta automaticamente). "
                        "mudar_assunto: cliente quer tratar de outro assunto que NÃO é "
                        "câmbio (ex.: limite de crédito) — preencha novo_assunto. "
                        "encerrar: cliente pediu para encerrar o atendimento. "
                        "conversar: qualquer outra resposta."
                    ),
                },
                "moeda_origem": {
                    "type": ["string", "null"],
                    "description": (
                        "Código ISO de 3 letras da moeda de origem, APENAS se o cliente "
                        "mencionou explicitamente uma moeda (nome ou código, ex.: "
                        "'dólar' -> USD, 'euro' -> EUR). Deixe null se o cliente ainda "
                        "não especificou nenhuma moeda — NUNCA preencha USD por conta "
                        "própria como suposição/padrão."
                    ),
                },
                "moeda_destino": {
                    "type": ["string", "null"],
                    "description": "Código ISO de 3 letras da moeda de destino (padrão BRL, caso o cliente não peça outra).",
                },
                "novo_assunto": {
                    "type": ["string", "null"],
                    "enum": ["credito", None],
                    "description": "Obrigatório quando acao=mudar_assunto: para qual assunto o cliente quer ir.",
                },
                "resposta_ao_cliente": {
                    "type": ["string", "null"],
                    "description": "Texto para o cliente quando acao=conversar.",
                },
            },
            "required": ["acao"],
        },
    },
}


def node_cambio(state: BancoAgilState) -> dict:
    try:
        return _executar(state)
    except Exception as exc:
        logger.exception("Erro inesperado no Agente de Câmbio: %s", exc)
        return {
            "messages": [AIMessage(content="Tivemos uma instabilidade ao consultar a cotação. Pode tentar novamente?")],
            "log_erros": state.get("log_erros", []) + [f"cambio: {exc}"],
            "continuar_mesmo_turno": False,
        }


def _executar(state: BancoAgilState) -> dict:
    system = SystemMessage(content=montar_prompt("cambio"))
    llm = get_llm().bind_tools([DECISAO_SCHEMA])
    resposta = invocar_com_recuperacao(llm, [system] + state["messages"], "decidir_acao_cambio")

    tool_calls = getattr(resposta, "tool_calls", None) or []
    decisao_call = next((c for c in tool_calls if c["name"] == "decidir_acao_cambio"), None)

    if not decisao_call:
        texto = resposta.content or "Qual moeda você gostaria de consultar?"
        return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

    args = decisao_call["args"]
    acao = args.get("acao", "conversar")

    if acao == "encerrar":
        return {
            "messages": [AIMessage(content="Tudo bem! Obrigado por falar com o Banco Ágil. Até breve!")],
            "encerrar_conversa": True,
            "agente_atual": "encerrado",
        }

    if acao == "mudar_assunto":
        novo_assunto = args.get("novo_assunto")
        if novo_assunto == "credito":
            return {
                "agente_atual": "credito",
                "messages": [],
                "continuar_mesmo_turno": True,
                "cambio_moeda_perguntada": False,
            }
        texto = "Você deseja consultar outra cotação, ou falar sobre seu crédito?"
        return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

    if acao == "consultar_cotacao":
        origem = args.get("moeda_origem")
        destino = args.get("moeda_destino") or "BRL"

        # Cliente ainda não especificou a moeda: mostra as opções em vez
        # de assumir dólar por padrão. Só faz isso uma vez por sessão de
        # câmbio, para não ficar repetindo a pergunta caso o cliente
        # continue vago mesmo depois de ver a lista.
        if not origem and not state.get("cambio_moeda_perguntada"):
            return {
                "messages": [AIMessage(content=MENU_MOEDAS)],
                "cambio_moeda_perguntada": True,
                "continuar_mesmo_turno": False,
            }

        origem = origem or "USD"
        resultado = consultar_cotacao.invoke({"moeda_origem": origem, "moeda_destino": destino})

        if not resultado.get("sucesso"):
            texto = resultado.get("erro", "Não consegui consultar essa cotação agora.")
            return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}

        texto = (
            f"1 {resultado['moeda_origem']} = {resultado['cotacao']:.4f} {resultado['moeda_destino']}. "
            "Posso ajudar com outra cotação, com seu crédito, ou encerramos por aqui?"
        )
        return {
            "messages": [AIMessage(content=texto)],
            "agente_atual": "triagem",
            "cambio_moeda_perguntada": False,
            "continuar_mesmo_turno": False,
        }

    texto = args.get("resposta_ao_cliente") or resposta.content or "Qual moeda você gostaria de consultar?"
    return {"messages": [AIMessage(content=texto)], "continuar_mesmo_turno": False}
