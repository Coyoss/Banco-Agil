"""
Cálculo do novo score de crédito com base na fórmula ponderada definida
no desafio. Fica isolado do agente para que o modelo NUNCA calcule o
score "de cabeça" (o que seria impreciso e não determinístico) — ele
sempre chama esta ferramenta com os dados coletados na entrevista.
"""
import logging

from langchain_core.tools import tool

logger = logging.getLogger("banco_agil.score_tools")

PESO_RENDA = 30
PESO_EMPREGO = {"formal": 300, "autônomo": 200, "autonomo": 200, "desempregado": 0}
PESO_DEPENDENTES = {"0": 100, "1": 80, "2": 60, "3+": 30}
PESO_DIVIDAS = {"sim": -100, "não": 100, "nao": 100}


def _normalizar_tipo_emprego(valor: str) -> str:
    v = (valor or "").strip().lower()
    if v in PESO_EMPREGO:
        return v
    if "form" in v or "clt" in v:
        return "formal"
    if "auton" in v or "autôn" in v or "freelan" in v or "conta própria" in v or "pj" in v:
        return "autônomo"
    if "desemp" in v or "sem emprego" in v or "sem renda" in v:
        return "desempregado"
    return "desempregado"  # fallback conservador


def _normalizar_dependentes(valor) -> str:
    try:
        n = int(valor)
        if n <= 0:
            return "0"
        if n == 1:
            return "1"
        if n == 2:
            return "2"
        return "3+"
    except (TypeError, ValueError):
        v = str(valor).strip().lower()
        if "3" in v or "mais" in v or "+" in v:
            return "3+"
        return "0"


def _normalizar_dividas(valor) -> str:
    v = str(valor).strip().lower()
    if v in ("sim", "s", "yes", "true", "possuo", "tenho"):
        return "sim"
    return "não"


@tool
def calcular_novo_score(
    renda_mensal: float,
    tipo_emprego: str,
    despesas_fixas_mensais: float,
    numero_dependentes: str,
    possui_dividas_ativas: str,
) -> dict:
    """
    Calcula o novo score de crédito (0 a 1000) a partir dos dados
    coletados na entrevista financeira, usando a fórmula ponderada oficial:

    score = (renda / (despesas + 1)) * peso_renda
            + peso_emprego[tipo_emprego]
            + peso_dependentes[num_dependentes]
            + peso_dividas[tem_dividas]

    tipo_emprego: "formal", "autônomo" ou "desempregado" (aceita variações).
    numero_dependentes: número (0, 1, 2, 3, 4...) — 3 ou mais conta como "3+".
    possui_dividas_ativas: "sim" ou "não" (aceita variações).
    """
    try:
        renda = float(renda_mensal)
        despesas = float(despesas_fixas_mensais)
    except (TypeError, ValueError):
        return {"sucesso": False, "erro": "Renda ou despesas informadas não são valores numéricos válidos."}

    if renda < 0 or despesas < 0:
        return {"sucesso": False, "erro": "Renda e despesas não podem ser negativas."}

    emprego = _normalizar_tipo_emprego(tipo_emprego)
    dependentes = _normalizar_dependentes(numero_dependentes)
    dividas = _normalizar_dividas(possui_dividas_ativas)

    try:
        bruto = (
            (renda / (despesas + 1)) * PESO_RENDA
            + PESO_EMPREGO[emprego]
            + PESO_DEPENDENTES[dependentes]
            + PESO_DIVIDAS[dividas]
        )
    except Exception as exc:
        logger.exception("Erro ao calcular score: %s", exc)
        return {"sucesso": False, "erro": "Não foi possível calcular o score no momento."}

    score_final = max(0.0, min(1000.0, round(bruto, 2)))

    return {
        "sucesso": True,
        "score": score_final,
        "detalhes": {
            "tipo_emprego_interpretado": emprego,
            "dependentes_interpretado": dependentes,
            "dividas_interpretado": dividas,
        },
    }
