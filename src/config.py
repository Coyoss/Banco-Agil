"""
Configurações centrais do Banco Ágil.
Carrega variáveis de ambiente e define caminhos usados em todo o projeto.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CLIENTES_CSV = DATA_DIR / "clientes.csv"
SCORE_LIMITE_CSV = DATA_DIR / "score_limite.csv"
SOLICITACOES_CSV = DATA_DIR / "solicitacoes_aumento_limite.csv"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Regras de negócio
MAX_TENTATIVAS_AUTENTICACAO = 3  # 1 tentativa inicial + 2 novas tentativas

# Colunas obrigatórias do arquivo de solicitações (contrato fixado pelo desafio)
SOLICITACOES_COLUNAS = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]
