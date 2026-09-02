"""
Ferramenta de consulta de cotação de moedas em tempo real.

Usa a API pública e gratuita da exchangerate-api.com (open access,
sem necessidade de chave), com timeout curto e tratamento de erro
completo — se a API estiver fora do ar, o agente é informado de forma
controlada em vez de quebrar a conversa.
"""
import logging

import requests
from langchain_core.tools import tool

logger = logging.getLogger("banco_agil.cambio_tools")

API_URL = "https://open.er-api.com/v6/latest/{moeda_base}"
TIMEOUT_SEGUNDOS = 6


@tool
def consultar_cotacao(moeda_origem: str = "USD", moeda_destino: str = "BRL") -> dict:
    """
    Consulta a cotação atual entre duas moedas (padrão: USD -> BRL).
    Use códigos ISO de 3 letras (USD, EUR, BRL, GBP, ARS, JPY etc.).
    Se o cliente falar "dólar", "euro" etc. converta para o código ISO
    antes de chamar esta ferramenta.
    """
    origem = (moeda_origem or "USD").strip().upper()
    destino = (moeda_destino or "BRL").strip().upper()

    try:
        resposta = requests.get(API_URL.format(moeda_base=origem), timeout=TIMEOUT_SEGUNDOS)
        resposta.raise_for_status()
        dados = resposta.json()

        if dados.get("result") != "success":
            return {
                "sucesso": False,
                "erro": "O serviço de cotação retornou um resultado inesperado. Tente novamente em instantes.",
            }

        taxas = dados.get("rates", {})
        if destino not in taxas:
            return {
                "sucesso": False,
                "erro": f"Não encontrei cotação para o código de moeda '{destino}'. Pode confirmar a moeda desejada?",
            }

        return {
            "sucesso": True,
            "moeda_origem": origem,
            "moeda_destino": destino,
            "cotacao": round(float(taxas[destino]), 4),
            "atualizado_em": dados.get("time_last_update_utc"),
        }

    except requests.exceptions.Timeout as exc:
        logger.exception("Timeout ao consultar câmbio: %s", exc)
        return {"sucesso": False, "erro": "O serviço de cotação demorou demais para responder. Pode tentar novamente?"}
    except requests.exceptions.RequestException as exc:
        logger.exception("Erro de rede ao consultar câmbio: %s", exc)
        return {"sucesso": False, "erro": "O serviço de cotação está indisponível no momento. Pode tentar novamente em instantes?"}
    except Exception as exc:
        logger.exception("Erro inesperado ao consultar câmbio: %s", exc)
        return {"sucesso": False, "erro": "Não foi possível obter a cotação agora. Pode tentar novamente?"}
