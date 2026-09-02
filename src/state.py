"""
Estado compartilhado entre todos os agentes no grafo do LangGraph.

Um único grafo com múltiplos nós (um por agente) compartilha este estado.
Isso é o que permite o redirecionamento IMPLÍCITO entre agentes: o cliente
sempre "fala com o mesmo atendente", mas o campo `agente_atual` decide,
por trás dos panos, qual nó do grafo processa a próxima mensagem.
"""
from typing import Annotated, Literal, Optional, TypedDict
from langgraph.graph.message import add_messages


AgentName = Literal["triagem", "credito", "entrevista", "cambio", "encerrado"]


class BancoAgilState(TypedDict, total=False):
    # Histórico de mensagens da conversa (usuário + assistente)
    messages: Annotated[list, add_messages]

    # Roteamento
    agente_atual: AgentName
    proximo_agente: Optional[AgentName]

    # Autenticação (Agente de Triagem)
    autenticado: bool
    tentativas_autenticacao: int
    cpf_cliente: Optional[str]
    nome_cliente: Optional[str]
    data_nascimento_cliente: Optional[str]  # ISO string

    # Dados do cliente carregados da base após autenticação
    limite_credito_atual: Optional[float]
    score_atual: Optional[float]

    # Contexto de crédito (Agente de Crédito)
    novo_limite_desejado: Optional[float]
    novo_limite_pendente_confirmacao: Optional[float]
    ultima_solicitacao_status: Optional[str]

    # Contexto de entrevista de crédito (Agente de Entrevista)
    entrevista_em_andamento: bool
    entrevista_respostas: dict
    solicitar_reavaliacao_credito: bool

    # Contexto de encerramento
    encerrar_conversa: bool

    # Contexto do Agente de Câmbio: controla se o menu de moedas
    # disponíveis já foi apresentado nesta sessão de câmbio, para não
    # assumir uma moeda padrão sem perguntar, mas também não ficar
    # perguntando repetidamente se o cliente continuar vago.
    cambio_moeda_perguntada: bool

    # Flag transiente: True quando um agente faz handoff interno e o
    # próximo nó deve processar a MESMA mensagem do usuário ainda nesta
    # execução (redirecionamento implícito, sem round-trip extra com o
    # cliente). Resetado para False a cada nova mensagem do usuário.
    continuar_mesmo_turno: bool

    # Registro leve de erros técnicos para eventual análise posterior
    log_erros: list
