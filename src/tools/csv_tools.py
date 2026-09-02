"""
Ferramentas (LangChain @tool) que leem/escrevem os arquivos CSV do banco.

Princípio seguido em todas as funções: NUNCA deixar uma exceção subir para
o agente sem tratamento. Toda função retorna um dicionário com pelo menos
a chave "sucesso" (bool). Em caso de falha, "erro" traz uma mensagem
segura para ser repassada ao cliente (sem stack trace, sem caminho de
arquivo, sem detalhes internos), e o problema completo é registrado via
`logging` para análise técnica posterior.
"""
import logging
import threading
from datetime import datetime, timezone

import pandas as pd
from langchain_core.tools import tool

from src.config import CLIENTES_CSV, SCORE_LIMITE_CSV, SOLICITACOES_CSV, SOLICITACOES_COLUNAS
from src.validators import normalize_cpf, normalize_date

logger = logging.getLogger("banco_agil.csv_tools")

# Lock simples para evitar condição de corrida em escrita concorrente
# (Streamlit roda multi-thread; múltiplas sessões podem escrever ao mesmo tempo).
_csv_lock = threading.Lock()


def _erro_generico(contexto: str, exc: Exception) -> dict:
    logger.exception("Erro em %s: %s", contexto, exc)
    return {
        "sucesso": False,
        "erro": (
            "Estamos com uma instabilidade técnica no momento para acessar essa "
            "informação. Você pode tentar novamente em instantes ou seguir para "
            "outro assunto."
        ),
    }


@tool
def autenticar_cliente(cpf: str, data_nascimento: str) -> dict:
    """
    Autentica um cliente contra a base clientes.csv.
    Aceita CPF e data de nascimento em QUALQUER formato de digitação
    (com ou sem pontuação/barras/traços) — a normalização é automática.

    Retorna um dicionário com:
      sucesso (bool), autenticado (bool),
      e, se autenticado: nome, cpf, limite_credito, score.
      Se não autenticado por CPF/data mal formados, "erro" explica o motivo
      (sem indicar qual dado especificamente está incorreto na base).
    """
    cpf_norm, erro_cpf = normalize_cpf(cpf)
    if erro_cpf:
        return {"sucesso": True, "autenticado": False, "motivo": "formato_invalido", "erro": erro_cpf}

    data_norm, erro_data = normalize_date(data_nascimento)
    if erro_data:
        return {"sucesso": True, "autenticado": False, "motivo": "formato_invalido", "erro": erro_data}

    try:
        df = pd.read_csv(CLIENTES_CSV, dtype={"cpf": str})
    except Exception as exc:
        return _erro_generico("autenticar_cliente/leitura", exc)

    try:
        df["cpf"] = df["cpf"].str.zfill(11)
        data_iso = data_norm.isoformat()
        linha = df[(df["cpf"] == cpf_norm) & (df["data_nascimento"] == data_iso)]

        if linha.empty:
            return {
                "sucesso": True,
                "autenticado": False,
                "motivo": "nao_encontrado",
                "erro": "CPF ou data de nascimento não conferem com nossos registros.",
            }

        registro = linha.iloc[0]
        return {
            "sucesso": True,
            "autenticado": True,
            "cpf": cpf_norm,
            "nome": str(registro["nome"]),
            "limite_credito": float(registro["limite_credito"]),
            "score": float(registro["score"]),
        }
    except Exception as exc:
        return _erro_generico("autenticar_cliente/processamento", exc)


@tool
def consultar_limite_credito(cpf: str) -> dict:
    """
    Consulta o limite de crédito atual e o score do cliente pelo CPF
    (o CPF já deve estar autenticado na conversa). Aceita CPF em qualquer
    formato de digitação.
    """
    cpf_norm, erro = normalize_cpf(cpf)
    if erro:
        return {"sucesso": False, "erro": erro}

    try:
        df = pd.read_csv(CLIENTES_CSV, dtype={"cpf": str})
        df["cpf"] = df["cpf"].str.zfill(11)
        linha = df[df["cpf"] == cpf_norm]
        if linha.empty:
            return {"sucesso": False, "erro": "Cliente não encontrado na base."}
        registro = linha.iloc[0]
        return {
            "sucesso": True,
            "limite_credito": float(registro["limite_credito"]),
            "score": float(registro["score"]),
        }
    except Exception as exc:
        return _erro_generico("consultar_limite_credito", exc)


@tool
def registrar_solicitacao_aumento_limite(cpf: str, limite_atual: float, novo_limite_solicitado: float) -> dict:
    """
    Registra uma solicitação de aumento de limite de crédito em
    solicitacoes_aumento_limite.csv e decide o status (aprovado/rejeitado)
    consultando score_limite.csv com base no score atual do cliente.

    Retorna: sucesso (bool), status_pedido ("aprovado"/"rejeitado"),
    limite_maximo_aprovavel_no_score (float), e erro (se sucesso=False).
    """
    cpf_norm, erro_cpf = normalize_cpf(cpf)
    if erro_cpf:
        return {"sucesso": False, "erro": erro_cpf}

    try:
        novo_limite_solicitado = float(novo_limite_solicitado)
        limite_atual = float(limite_atual)
    except (TypeError, ValueError):
        return {"sucesso": False, "erro": "O valor de limite informado não é um número válido."}

    if novo_limite_solicitado <= 0:
        return {"sucesso": False, "erro": "O novo limite solicitado precisa ser maior que zero."}

    try:
        clientes = pd.read_csv(CLIENTES_CSV, dtype={"cpf": str})
        clientes["cpf"] = clientes["cpf"].str.zfill(11)
        linha_cliente = clientes[clientes["cpf"] == cpf_norm]
        if linha_cliente.empty:
            return {"sucesso": False, "erro": "Cliente não encontrado na base."}
        score = float(linha_cliente.iloc[0]["score"])

        tabela_score = pd.read_csv(SCORE_LIMITE_CSV)
        tabela_score = tabela_score.sort_values("score_minimo")
        faixa = tabela_score[tabela_score["score_minimo"] <= score]
        limite_maximo_aprovavel = float(faixa.iloc[-1]["limite_maximo_aprovavel"]) if not faixa.empty else 0.0

        status = "aprovado" if novo_limite_solicitado <= limite_maximo_aprovavel else "rejeitado"

        with _csv_lock:
            nova_linha = pd.DataFrame([{
                "cpf_cliente": cpf_norm,
                "data_hora_solicitacao": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "limite_atual": round(limite_atual, 2),
                "novo_limite_solicitado": round(novo_limite_solicitado, 2),
                "status_pedido": status,
            }], columns=SOLICITACOES_COLUNAS)

            try:
                existentes = pd.read_csv(SOLICITACOES_CSV)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                existentes = pd.DataFrame(columns=SOLICITACOES_COLUNAS)

            atualizado = pd.concat([existentes, nova_linha], ignore_index=True)
            atualizado.to_csv(SOLICITACOES_CSV, index=False)

            if status == "aprovado":
                clientes.loc[clientes["cpf"] == cpf_norm, "limite_credito"] = round(novo_limite_solicitado, 2)
                clientes.to_csv(CLIENTES_CSV, index=False)

        return {
            "sucesso": True,
            "status_pedido": status,
            "limite_maximo_aprovavel_no_score": limite_maximo_aprovavel,
        }
    except Exception as exc:
        return _erro_generico("registrar_solicitacao_aumento_limite", exc)


@tool
def atualizar_score_cliente(cpf: str, novo_score: float) -> dict:
    """
    Atualiza o score de crédito do cliente na base clientes.csv após uma
    entrevista de crédito.
    """
    cpf_norm, erro_cpf = normalize_cpf(cpf)
    if erro_cpf:
        return {"sucesso": False, "erro": erro_cpf}

    try:
        novo_score = max(0.0, min(1000.0, float(novo_score)))
    except (TypeError, ValueError):
        return {"sucesso": False, "erro": "Score calculado é inválido."}

    try:
        with _csv_lock:
            df = pd.read_csv(CLIENTES_CSV, dtype={"cpf": str})
            df["cpf"] = df["cpf"].str.zfill(11)
            if not (df["cpf"] == cpf_norm).any():
                return {"sucesso": False, "erro": "Cliente não encontrado na base."}
            # A coluna "score" pode ter sido lida como int64 quando todos os
            # valores gravados até agora eram inteiros. Forçamos float64 antes
            # de atribuir, para o pandas não recusar um score com casas
            # decimais (ex.: 757.95) por incompatibilidade de dtype.
            df["score"] = df["score"].astype("float64")
            df.loc[df["cpf"] == cpf_norm, "score"] = round(novo_score, 2)
            df.to_csv(CLIENTES_CSV, index=False)
        return {"sucesso": True, "novo_score": novo_score}
    except Exception as exc:
        return _erro_generico("atualizar_score_cliente", exc)
