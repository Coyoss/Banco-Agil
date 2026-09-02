"""
Nó do Agente de Triagem no grafo LangGraph.

Responsável por autenticar o cliente (CPF + data de nascimento, em
qualquer formato) e, uma vez autenticado, identificar o assunto e
encaminhar (de forma implícita) para o agente correto.
"""
import json
import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.config import GROQ_MODEL, MAX_TENTATIVAS_AUTENTICACAO
from src.llm import get_llm, invocar_com_recuperacao
from src.prompts import montar_prompt
from src.state import BancoAgilState
from src.tools.csv_tools import autenticar_cliente

logger = logging.getLogger("banco_agil.agents.triagem")

# Menu de assuntos reaproveitado em toda mensagem que oferece as opções ao
# cliente, para manter o texto (e a numeração) sempre idênticos — isso ajuda
# o modelo a reconhecer respostas como "1", "2" ou "3" mais adiante.
MENU_ASSUNTOS = (
    "Como posso te ajudar hoje?\n"
    "1. Consultar meu limite de crédito\n"
    "2. Aumentar meu limite de crédito\n"
    "3. Consultar cotação de moeda"
)

_TOOLS = [autenticar_cliente]

# Ferramenta "virtual" que o modelo usa para sinalizar decisões de fluxo
# que não envolvem acesso a dados (roteamento e encerramento).
ROTEAMENTO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "definir_proximo_passo",
        "description": (
            "Define o que fazer a seguir após identificar o assunto do cliente "
            "autenticado, ou sinaliza que o cliente pediu para encerrar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proximo_agente": {
                    "type": "string",
                    "enum": ["credito", "cambio", "nenhum"],
                    "description": "Para qual assunto encaminhar, ou 'nenhum' se ainda não souber.",
                },
                "encerrar": {
                    "type": "boolean",
                    "description": "true se o cliente pediu para encerrar o atendimento.",
                },
            },
            "required": ["proximo_agente", "encerrar"],
        },
    },
}


def _llm_com_ferramentas():
    llm = get_llm()
    return llm.bind_tools(_TOOLS + [ROTEAMENTO_SCHEMA])


def node_triagem(state: BancoAgilState) -> dict:
    try:
        return _executar(state)
    except Exception as exc:
        logger.exception("Erro inesperado no Agente de Triagem: %s", exc)
        return {
            "messages": [AIMessage(content=(
                "Tivemos uma instabilidade técnica por aqui. Pode tentar novamente "
                "em instantes? Se preferir, podemos encerrar o atendimento."
            ))],
            "log_erros": state.get("log_erros", []) + [f"triagem: {exc}"],
            "continuar_mesmo_turno": False,
        }


def _executar(state: BancoAgilState) -> dict:
    if state.get("autenticado"):
        # Já autenticado: apenas identifica o assunto e roteia.
        return _identificar_assunto_e_rotear(state)

    tentativas = state.get("tentativas_autenticacao", 0)

    system = SystemMessage(content=montar_prompt("triagem"))
    llm = _llm_com_ferramentas()
    resposta = llm.invoke([system] + state["messages"])

    tool_calls = getattr(resposta, "tool_calls", None) or []
    autenticacao_call = next((c for c in tool_calls if c["name"] == "autenticar_cliente"), None)

    novo_estado: dict = {}

    if autenticacao_call:
        resultado = autenticar_cliente.invoke(autenticacao_call["args"])

        if resultado.get("autenticado"):
            novo_estado.update({
                "autenticado": True,
                "cpf_cliente": resultado["cpf"],
                "nome_cliente": resultado["nome"],
                "limite_credito_atual": resultado["limite_credito"],
                "score_atual": resultado["score"],
                "tentativas_autenticacao": tentativas,
            })
            texto = (
                f"Perfeito, {resultado['nome'].split()[0]}! Autenticação confirmada.\n\n"
                f"{MENU_ASSUNTOS}"
            )
            novo_estado["messages"] = [AIMessage(content=texto)]
            novo_estado["agente_atual"] = "triagem"
            novo_estado["continuar_mesmo_turno"] = False
            return novo_estado

        # Falha de autenticação (formato inválido OU não encontrado)
        tentativas += 1
        novo_estado["tentativas_autenticacao"] = tentativas

        if tentativas >= MAX_TENTATIVAS_AUTENTICACAO:
            texto = (
                "Não consegui confirmar seus dados após algumas tentativas. Para sua "
                "segurança, vou encerrar este atendimento por aqui. Você pode iniciar "
                "uma nova conversa quando quiser tentar novamente. Obrigado pela "
                "compreensão!"
            )
            novo_estado["messages"] = [AIMessage(content=texto)]
            novo_estado["encerrar_conversa"] = True
            novo_estado["agente_atual"] = "encerrado"
            return novo_estado

        restantes = MAX_TENTATIVAS_AUTENTICACAO - tentativas
        motivo = resultado.get("erro") if resultado.get("motivo") == "formato_invalido" else (
            "Não consegui confirmar seus dados. Pode conferir o CPF e a data de "
            "nascimento e enviar novamente?"
        )
        texto = f"{motivo} Você ainda tem {restantes} tentativa(s)."
        novo_estado["messages"] = [AIMessage(content=texto)]
        novo_estado["agente_atual"] = "triagem"
        novo_estado["continuar_mesmo_turno"] = False
        return novo_estado

    # Modelo ainda coletando CPF/data (nenhuma tool call de autenticação ainda)
    conteudo = resposta.content or "Pode me informar seu CPF e sua data de nascimento, por favor?"
    novo_estado["messages"] = [AIMessage(content=conteudo)]
    novo_estado["agente_atual"] = "triagem"
    novo_estado["continuar_mesmo_turno"] = False
    return novo_estado


def _identificar_assunto_e_rotear(state: BancoAgilState) -> dict:
    system = SystemMessage(content=montar_prompt("triagem") + (
        "\n\nO cliente JÁ está autenticado. Sua única tarefa agora é identificar, "
        "pela última mensagem, se o assunto é crédito (consulta/aumento de limite) "
        "ou câmbio (cotação de moeda), ou se ainda não ficou claro. Sempre chame a "
        "ferramenta definir_proximo_passo para registrar sua decisão."
    ))
    llm = get_llm().bind_tools([ROTEAMENTO_SCHEMA])
    resposta = invocar_com_recuperacao(llm, [system] + state["messages"], "definir_proximo_passo")

    tool_calls = getattr(resposta, "tool_calls", None) or []
    decisao_call = next((c for c in tool_calls if c["name"] == "definir_proximo_passo"), None)

    if not decisao_call:
        texto = resposta.content or MENU_ASSUNTOS
        return {"messages": [AIMessage(content=texto)], "agente_atual": "triagem", "continuar_mesmo_turno": False}

    args = decisao_call["args"]

    if args.get("encerrar"):
        return {
            "messages": [AIMessage(content="Tudo bem! Obrigado por falar com o Banco Ágil. Até breve!")],
            "encerrar_conversa": True,
            "agente_atual": "encerrado",
        }

    proximo = args.get("proximo_agente", "nenhum")
    if proximo in ("credito", "cambio"):
        atualizacao = {
            "agente_atual": proximo,
            "proximo_agente": None,
            "messages": [],
            "continuar_mesmo_turno": True,
        }
        if proximo == "cambio":
            atualizacao["cambio_moeda_perguntada"] = False
        return atualizacao

    return {"messages": [AIMessage(content=MENU_ASSUNTOS)], "agente_atual": "triagem", "continuar_mesmo_turno": False}
