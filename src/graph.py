"""
Grafo LangGraph do Banco Ágil.

Arquitetura de roteamento:
- Cada agente é um nó. O estado carrega `agente_atual`, persistido entre
  turnos (mensagens) da conversa.
- O PONTO DE ENTRADA é condicional: a cada nova mensagem do cliente, o
  grafo entra diretamente no nó do agente que estava "no controle" na
  última interação (ou em `triagem` se ainda não autenticado).
- Handoff IMPLÍCITO dentro do mesmo turno: quando um agente decide passar
  a conversa para outro (ex.: Triagem -> Crédito ao identificar o
  assunto, ou Crédito -> Entrevista ao aceitar a entrevista), ele marca
  `continuar_mesmo_turno=True` e não emite mensagem própria. O grafo então
  segue direto para o próximo nó, AINDA dentro da mesma execução, para que
  o cliente receba uma resposta completa sem perceber a transição.
- Quando um agente emite uma mensagem para o cliente (fim de turno) ou o
  atendimento é encerrado, o grafo termina (END) e aguarda a próxima
  mensagem do cliente.
"""
from langgraph.graph import StateGraph, START, END

from src.agents.cambio import node_cambio
from src.agents.credito import node_credito
from src.agents.entrevista import node_entrevista
from src.agents.triagem import node_triagem
from src.state import BancoAgilState


def _entry_point(state: BancoAgilState) -> str:
    if not state.get("autenticado"):
        return "triagem"
    agente = state.get("agente_atual") or "triagem"
    if agente in ("credito", "cambio", "entrevista"):
        return agente
    return "triagem"


def _route_after(state: BancoAgilState) -> str:
    if state.get("encerrar_conversa"):
        return END
    if state.get("continuar_mesmo_turno"):
        proximo = state.get("agente_atual") or "triagem"
        if proximo in ("triagem", "credito", "cambio", "entrevista"):
            return proximo
    return END


def build_graph():
    workflow = StateGraph(BancoAgilState)

    workflow.add_node("triagem", node_triagem)
    workflow.add_node("credito", node_credito)
    workflow.add_node("entrevista", node_entrevista)
    workflow.add_node("cambio", node_cambio)

    entrada = {"triagem": "triagem", "credito": "credito", "cambio": "cambio", "entrevista": "entrevista"}
    workflow.add_conditional_edges(START, _entry_point, entrada)

    destinos = {"triagem": "triagem", "credito": "credito", "cambio": "cambio", "entrevista": "entrevista", END: END}
    workflow.add_conditional_edges("triagem", _route_after, destinos)
    workflow.add_conditional_edges("credito", _route_after, destinos)
    workflow.add_conditional_edges("entrevista", _route_after, destinos)
    workflow.add_conditional_edges("cambio", _route_after, destinos)

    return workflow.compile()


_grafo_compilado = None


def get_graph():
    global _grafo_compilado
    if _grafo_compilado is None:
        _grafo_compilado = build_graph()
    return _grafo_compilado
