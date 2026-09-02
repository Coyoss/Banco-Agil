"""
Validadores tolerantes a formato para CPF e data de nascimento.

Objetivo: o cliente pode digitar o CPF e a data de qualquer jeito
(com ou sem pontuação, com barras, traços, pontos, por extenso etc.)
que o sistema normaliza para um formato canônico antes de qualquer
comparação com a base de dados.

Nenhuma função aqui levanta exceção para entrada "errada" do usuário:
todas retornam (valor_normalizado_ou_None, mensagem_de_erro_ou_None),
para que o agente sempre consiga responder de forma controlada.
"""
import re
from datetime import datetime, date
from dateutil import parser as dateparser
from dateutil.parser import ParserError


def _only_digits(texto: str) -> str:
    return re.sub(r"\D", "", texto or "")


def normalize_cpf(cpf_bruto: str):
    """
    Aceita CPF com ou sem pontuação (000.000.000-00, 000 000 000 00,
    00000000000 etc.) e retorna a versão normalizada de 11 dígitos.

    Retorna (cpf_normalizado, None) em caso de sucesso,
    ou (None, mensagem_de_erro) em caso de falha.
    """
    if not cpf_bruto or not str(cpf_bruto).strip():
        return None, "CPF não informado."

    digitos = _only_digits(str(cpf_bruto))

    if len(digitos) != 11:
        return None, (
            f"O CPF informado tem {len(digitos)} dígito(s), mas um CPF válido "
            "possui 11 dígitos. Pode conferir e enviar novamente?"
        )

    if digitos == digitos[0] * 11:
        return None, "Esse CPF não parece válido (todos os dígitos são iguais)."

    if not _validar_digitos_verificadores_cpf(digitos):
        return None, "Esse CPF não é válido (dígitos verificadores não conferem)."

    return digitos, None


def _validar_digitos_verificadores_cpf(cpf: str) -> bool:
    def calc_digito(parcial: str, peso_inicial: int) -> int:
        soma = sum(int(d) * peso for d, peso in zip(parcial, range(peso_inicial, 1, -1)))
        resto = (soma * 10) % 11
        return 0 if resto == 10 else resto

    d1 = calc_digito(cpf[:9], 10)
    d2 = calc_digito(cpf[:9] + str(d1), 11)
    return cpf[-2:] == f"{d1}{d2}"


def format_cpf_display(cpf_normalizado: str) -> str:
    """Formata 11 dígitos como 000.000.000-00 apenas para exibição."""
    if not cpf_normalizado or len(cpf_normalizado) != 11:
        return cpf_normalizado or ""
    c = cpf_normalizado
    return f"{c[0:3]}.{c[3:6]}.{c[6:9]}-{c[9:11]}"


def normalize_date(data_bruta: str):
    """
    Aceita data em praticamente qualquer formato textual comum:
    12/04/1990, 12-04-1990, 12.04.1990, 1990-04-12, 12 de abril de 1990,
    12 abril 1990, 04/12/1990 (interpretado dia-primeiro por padrão), etc.

    Retorna (date_object, None) em caso de sucesso,
    ou (None, mensagem_de_erro) em caso de falha.
    """
    if not data_bruta or not str(data_bruta).strip():
        return None, "Data de nascimento não informada."

    texto = str(data_bruta).strip()

    # Meses por extenso em português não são reconhecidos nativamente pelo
    # dateutil (que espera inglês) -> traduzimos antes do parse.
    texto_traduzido = _traduzir_meses_pt(texto)

    # Caso já venha em ISO (YYYY-MM-DD ou YYYY/MM/DD), tenta primeiro sem
    # ambiguidade de dayfirst.
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", texto)
    if m:
        try:
            ano, mes, dia = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dt = date(ano, mes, dia)
            return _validar_intervalo(dt)
        except ValueError:
            return None, "Essa data não existe. Pode conferir e enviar novamente?"

    try:
        dt = dateparser.parse(texto_traduzido, dayfirst=True, fuzzy=True).date()
    except (ParserError, ValueError, OverflowError, TypeError):
        return None, (
            "Não consegui entender essa data. Pode informar, por exemplo, "
            "12/04/1990 ou 1990-04-12?"
        )

    return _validar_intervalo(dt)


def _validar_intervalo(dt: date):
    hoje = date.today()
    if dt > hoje:
        return None, "Essa data de nascimento está no futuro. Pode conferir e enviar novamente?"
    idade = (hoje - dt).days / 365.25
    if idade > 120:
        return None, "Essa data de nascimento resultaria em uma idade improvável. Pode conferir?"
    return dt, None


def parse_numero(texto: str):
    """
    Extrai um número (float) de textos livres em português, tolerando
    formatos como "3000", "3.000,00", "R$ 3000", "3 mil", "3.5 mil" etc.

    Retorna (valor, None) ou (None, mensagem_de_erro).
    """
    if texto is None or not str(texto).strip():
        return None, "Não entendi o valor informado."

    t = str(texto).strip().lower()
    multiplicador = 1.0
    if "mil" in t:
        multiplicador = 1000.0
        t = t.replace("mil", "")

    t = t.replace("r$", "").replace("reais", "").strip()
    t = re.sub(r"[^0-9,.\-]", "", t)

    if not t:
        return None, "Não consegui identificar um número nessa mensagem. Pode informar apenas o valor?"

    # Trata tanto "3.000,50" (BR) quanto "3000.50" (US)
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".")

    try:
        valor = float(t) * multiplicador
        if valor < 0:
            return None, "O valor não pode ser negativo."
        return valor, None
    except ValueError:
        return None, "Não consegui identificar um número válido nessa mensagem. Pode reenviar apenas o valor?"


def format_date_display(dt: date) -> str:
    if not dt:
        return ""
    return dt.strftime("%d/%m/%Y")


_MESES_PT = {
    "janeiro": "january", "fevereiro": "february", "março": "march", "marco": "march",
    "abril": "april", "maio": "may", "junho": "june", "julho": "july",
    "agosto": "august", "setembro": "september", "outubro": "october",
    "novembro": "november", "dezembro": "december",
}


def _traduzir_meses_pt(texto: str) -> str:
    texto_lower = texto.lower()
    for pt, en in _MESES_PT.items():
        if pt in texto_lower:
            texto_lower = texto_lower.replace(pt, en)
    return texto_lower
