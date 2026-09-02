# 🏦 Banco Ágil — Atendimento Bancário com Agentes de IA

Sistema de atendimento ao cliente do **Banco Ágil**, um banco digital fictício,
construído com múltiplos agentes de IA especializados orquestrados por um
único grafo **LangGraph** e **Groq-AI**, com interface de teste em **Streamlit**.

Desenvolvido como solução do Desafio Técnico: Agente Bancário Inteligente.

---

## 📋 Visão Geral

O sistema simula um atendimento bancário completo através de quatro agentes
especializados que, para o cliente, se comportam como **um único atendente**
toda transição entre agentes é implícita e invisível para quem está do
outro lado da conversa:

| Agente | Responsabilidade |
|---|---|
| **Triagem** | Recepciona o cliente, autentica (CPF + data de nascimento) e identifica o assunto |
| **Crédito** | Consulta limite/score disponível e processa solicitações de aumento de limite (com confirmação explícita do cliente) |
| **Entrevista de Crédito** | Conduz uma entrevista financeira e recalcula o score de crédito |
| **Câmbio** | Consulta cotações de moedas em tempo real |

Além dos requisitos obrigatórios do desafio, o projeto implementa as
seguintes melhorias adicionais:

1. **Envio de mensagem bloqueado durante o processamento** — o campo de
   texto só é liberado depois que o agente termina de responder, com um
   indicador visual de "digitando...".
2. **Aceitação de CPF e data em qualquer formato** — com ou sem pontuação,
   por extenso, em qualquer ordem de separadores comuns, com validação real
   de dígito verificador do CPF.
3. **Prompt base comum a todos os agentes + prompt específico por agente**,
   cobrindo segurança, tratamento de erros e falhas internas — reforçado
   também em código, não apenas em texto de prompt.
4. **Confirmação explícita antes de qualquer alteração de limite** — o
   Agente de Crédito nunca registra ou aplica um aumento de limite
   automaticamente: ele sempre resume o pedido e aguarda o cliente confirmar
   antes de gravar qualquer coisa em disco.
5. **Troca de assunto no meio da conversa** — se o cliente estiver falando
   de crédito e perguntar sobre câmbio (ou vice-versa), o agente atual
   reconhece isso e faz o encaminhamento implícito, em vez de responder fora
   do seu escopo ou recusar o pedido.

---

## 🏗 Arquitetura do Sistema

### Visão geral do grafo

O sistema é um único **grafo LangGraph** (`src/graph.py`) com quatro nós —
um por agente — e um **estado compartilhado** (`src/state.py`) que carrega
todo o contexto da conversa (autenticação, dados do cliente, histórico de
mensagens, solicitações pendentes de confirmação, etc).

```
                     ┌────────────┐
      novo cliente → │  Triagem   │ ← ponto de entrada condicional
                     └─────┬──────┘   (decide, a cada mensagem, qual nó
                           │           processa com base no estado salvo)
              assunto identificado
                     ┌─────┴──────┐
                     ▼            ▼
               ┌──────────┐  ┌──────────┐
               │ Crédito  │◄─┤  Câmbio  │
               └────┬─────┘  └────┬─────┘
                     └──────┬─────┘
              troca de assunto em qualquer direção
           pedido rejeitado
                     ▼
             ┌───────────────┐
             │  Entrevista    │
             │  de Crédito    │
             └───────┬────────┘
                      │  (devolve ao Crédito p/ nova análise automática)
                      ▼
               ┌──────────┐
               │ Crédito  │
               └──────────┘
```

### Redirecionamento implícito (o "pulo do gato")

O requisito mais delicado do desafio é que o redirecionamento entre agentes
seja **implícito** — o cliente nunca deve perceber uma "troca de atendente".
Isso foi resolvido com dois mecanismos:

- **Ponto de entrada condicional**: a cada nova mensagem do cliente, o
  grafo entra diretamente no nó do agente que estava em controle na última
  interação (`agente_atual`, persistido no estado da sessão).
- **Handoff dentro do mesmo turno**: quando um agente decide passar a
  conversa para outro (ex.: Triagem → Crédito ao identificar o assunto,
  Crédito → Entrevista ao aceitar a entrevista, ou Crédito ↔ Câmbio ao
  mudar de assunto), ele marca `continuar_mesmo_turno=True` e não emite
  mensagem própria. O grafo então segue **imediatamente** para o próximo
  nó, ainda dentro da mesma execução (`graph.invoke`), de forma que o
  cliente recebe uma resposta completa e natural, sem round-trips extras
  nem "avisos de transferência".

Esse mecanismo é usado, por exemplo, quando:
- O cliente já autenticado pede para aumentar o limite → Triagem identifica
  o assunto e já entrega a mensagem ao Agente de Crédito na mesma resposta.
- O cliente está no meio de uma conversa sobre crédito e pergunta a cotação
  do dólar → o Agente de Crédito reconhece a ação `mudar_assunto` e entrega
  a mensagem ao Agente de Câmbio na mesma resposta (e vice-versa).
- Um pedido é rejeitado e o cliente aceita a entrevista → Crédito entrega
  para Entrevista, que já faz a primeira pergunta na mesma resposta.
- A entrevista termina → Entrevista recalcula o score, atualiza a base e
  devolve automaticamente ao Crédito, que reavalia o pedido pendente sem
  precisar que o cliente peça novamente.

### Confirmação explícita de aumento de limite

Diferente de uma aprovação/rejeição automática assim que o cliente informa
o valor desejado, o fluxo real é em duas etapas:

1. Cliente informa o novo limite desejado → o agente **apenas apresenta um
   resumo** ("de R$ X para R$ Y") e pergunta se pode seguir. Nada é gravado
   em disco ainda — o valor fica em `novo_limite_pendente_confirmacao` no
   estado da conversa.
2. Só depois que o cliente confirma explicitamente (`confirmar_solicitacao`)
   a ferramenta de registro é de fato chamada, gravando o pedido em
   `solicitacoes_aumento_limite.csv` e aplicando a aprovação/rejeição. Se o
   cliente recusar (`cancelar_solicitacao`), nada é registrado.

### Como os dados são manipulados

| Arquivo | Uso |
|---|---|
| `data/clientes.csv` | Base de clientes (cpf, nome, data de nascimento, limite, score) — lida na autenticação e nas consultas, atualizada após aprovação de aumento de limite e após a entrevista de crédito |
| `data/score_limite.csv` | Faixas de score → limite máximo aprovável, usada para decidir aprovação/rejeição |
| `data/solicitacoes_aumento_limite.csv` | Log de todas as solicitações de aumento de limite **confirmadas** pelo cliente, com timestamp ISO 8601 e status |

Todo acesso a esses arquivos passa por funções em `src/tools/csv_tools.py`,
expostas aos agentes como *tools* do LangChain — o modelo nunca lê/escreve
os arquivos diretamente, apenas decide **quando** chamar a ferramenta certa.
Um lock (`threading.Lock`) protege escritas concorrentes, já que o Streamlit
pode atender múltiplas sessões em threads simultâneas.

### Estrutura de pastas

```
banco-agil/
├── app.py                     # Interface Streamlit
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml             # Tema base do Streamlit
├── data/
│   ├── clientes.csv
│   ├── score_limite.csv
│   └── solicitacoes_aumento_limite.csv
└── src/
    ├── config.py               # Configurações e caminhos centrais
    ├── state.py                 # Estado compartilhado do grafo (TypedDict)
    ├── graph.py                  # Construção do grafo LangGraph
    ├── llm.py                     # Inicialização do LLM (Groq)
    ├── prompts.py                  # Prompt base + prompt específico por agente
    ├── validators.py                # Normalização flexível de CPF/data/números
    ├── agents/
    │   ├── triagem.py                # Autenticação + roteamento
    │   ├── credito.py                 # Consulta, score e aumento de limite (com confirmação)
    │   ├── entrevista.py               # Entrevista de crédito
    │   └── cambio.py                    # Cotação de moedas
    └── tools/
        ├── csv_tools.py                 # Acesso a clientes.csv / solicitações
        ├── score_tools.py                # Cálculo determinístico do score
        └── cambio_tools.py                # Consulta de cotação (API externa)
```

---

## ✅ Funcionalidades Implementadas

**Obrigatórias (conforme desafio):**
- Agente de Triagem com autenticação (CPF + data de nascimento) contra base
  CSV, com até 2 novas tentativas após falha e encerramento educado na 3ª.
- Agente de Crédito: consulta de limite/score e fluxo completo de
  solicitação de aumento, com registro em CSV (`cpf_cliente`,
  `data_hora_solicitacao` em ISO 8601, `limite_atual`,
  `novo_limite_solicitado`, `status_pedido`) e decisão automática de
  aprovação/rejeição com base em `score_limite.csv`.
- Oferta automática de encaminhamento para a Entrevista de Crédito quando um
  pedido é rejeitado.
- Agente de Entrevista de Crédito: coleta conversacional (uma pergunta por
  vez) de renda, tipo de emprego, despesas, dependentes e dívidas; cálculo
  do novo score pela fórmula ponderada oficial; atualização da base; retorno
  automático ao Crédito para nova análise.
- Agente de Câmbio: cotação em tempo real via API pública, com encerramento
  amigável do assunto específico.
- Encerramento a qualquer momento a pedido do cliente.
- Redirecionamento implícito entre todos os agentes.
- Tratamento de erros e exceções em toda chamada de ferramenta (CSV, API
  externa), sem expor detalhes técnicos ao cliente.
- Interface Streamlit funcional para testes.

**Adicionais (solicitadas para esta versão):**
1. Campo de mensagem bloqueado enquanto o agente processa a resposta
   (indicador de "digitando..." com bolhas animadas).
2. Aceitação de CPF e data de nascimento em qualquer formato de digitação,
   com validação real de dígito verificador do CPF.
3. Interface redesenhada, moderna, com **botão próprio de alternância entre
   tema claro e escuro** que força até sidebar, cabeçalho e barra de
   digitação a seguirem a paleta escolhida (não depende do menu nativo do
   Streamlit).
4. Prompt base comum (regras de segurança, tom, tratamento de erros) + prompt
   específico por agente, com reforço equivalente em código.
5. Confirmação explícita do cliente antes de qualquer aumento de limite ser
   de fato registrado/aplicado — nunca automático.
6. Troca de assunto (crédito ↔ câmbio) reconhecida no meio de qualquer
   fluxo, com handoff implícito, em vez de recusa ou resposta fora do
   escopo.
7. Consulta de score de crédito disponível como pergunta direta ao Agente
   de Crédito, sem precisar passar pela Entrevista.
8. Painel lateral com CPFs e datas de nascimento de clientes de teste, para
   agilizar a validação manual do fluxo completo.

---

## 🧩 Desafios Enfrentados e Como Foram Resolvidos

- **Handoff implícito sem round-trip extra**: resolvido com o padrão
  `continuar_mesmo_turno` descrito acima — evita tanto uma mensagem de
  "te transferindo..." quanto a necessidade de o cliente repetir a
  pergunta para o "próximo agente".
- **Evitar loop infinito no grafo**: como o roteamento depende de uma flag
  transiente (`continuar_mesmo_turno`) persistida manualmente entre
  invocações (não usamos checkpointer do LangGraph), havia risco de a flag
  "vazar" de um turno para o outro e causar reprocessamento indevido. Cada
  nó agora **sempre** define explicitamente essa flag em todo retorno —
  `True` apenas nos handoffs internos, `False` em toda resposta final ao
  cliente — eliminando o risco de loop.
- **Reavaliação automática pós-entrevista sem confundir o LLM**: depois que
  a entrevista termina, a última mensagem do histórico é a resposta à
  pergunta sobre dívidas — não um novo pedido do cliente. Deixar o LLM
  "decidir" a partir dessa mensagem geraria respostas erráticas. A solução
  foi tratar esse caso como um fluxo **determinístico em código** (flag
  `solicitar_reavaliacao_credito`), sem nova chamada ao modelo.
- **Perguntas fora do roteiro previsto geravam respostas inventadas**: no
  início, ações não mapeadas explicitamente no schema de decisão de um
  agente (ex.: "qual meu score?" enquanto conversando com o Agente de
  Crédito) caíam num fallback genérico onde o modelo podia "alucinar" uma
  recusa. A solução foi mapear explicitamente as ações `consultar_score` e
  `mudar_assunto` nos schemas de Crédito e Câmbio, em vez de depender do
  fallback livre para tudo que não foi previsto.
- **Evitar alteração automática de dados sensíveis**: inicialmente, assim
  que o cliente informava o novo limite desejado, a solicitação já era
  registrada e avaliada. Isso foi ajustado para um fluxo de duas etapas —
  apresentar o resumo do pedido e só registrar após confirmação explícita
  (`confirmar_solicitacao`/`cancelar_solicitacao`) — usando um campo de
  estado (`novo_limite_pendente_confirmacao`) para não depender da memória
  do LLM entre uma mensagem e outra.
- **Aceitar qualquer formato de CPF/data**: implementado com extração de
  dígitos + validação de dígito verificador para CPF, e `dateutil.parser`
  com tradução de meses por extenso em português para datas, com validação
  de intervalo plausível (não aceita datas futuras ou idades > 120 anos).
- **Nunca deixar o cliente ver erro técnico**: toda função de acesso a CSV
  ou API externa retorna um dicionário padronizado (`sucesso`/`erro`) e
  registra a exceção real via `logging`, sem propagar stack traces —
  reforçado tanto em `prompts.py` (regra de comportamento) quanto em código
  (`_erro_generico` em `csv_tools.py`, blocos `try/except` em cada nó).

---

## 🛠 Escolhas Técnicas e Justificativas

- **LangGraph** para orquestração: modela naturalmente o conceito de
  "estado compartilhado + múltiplos especialistas" exigido pelo desafio,
  com roteamento condicional explícito (mais previsível e auditável do que
  deixar um único agente "genérico" decidir tudo via prompt).
- **Groq (OpenAI )** como LLM: free tier generoso, latência baixa
  (importante para uma experiência de chat fluida) e suporte a *tool
  calling*, usado extensivamente para decisões estruturadas dos agentes.
- **Ações críticas via *tool calling* estruturado, não texto livre**: em vez
  de pedir ao modelo para "decidir e explicar em texto", cada agente usa uma
  ferramenta de decisão com `enum` fixo (ex.: `decidir_acao_credito`, com
  ações como `consultar_score`, `solicitar_aumento`,
  `confirmar_solicitacao`, `mudar_assunto`). Isso reduz ambiguidade e torna
  o comportamento do agente testável e previsível.
- **Cálculos de negócio nunca feitos pelo LLM**: o score de crédito e a
  aprovação/rejeição de limite são sempre calculados em código Python
  determinístico (`score_tools.py`, `csv_tools.py`), nunca "estimados" pelo
  modelo — essencial para um sistema financeiro.
- **Alteração de dados só após confirmação explícita**: o registro de uma
  solicitação de aumento de limite (e a consequente aprovação/rejeição)
  nunca acontece na mesma mensagem em que o cliente informa o valor —
  sempre há uma etapa intermediária de confirmação, controlada por código
  (não apenas por instrução de prompt), para reduzir o risco de uma
  alteração indesejada por ambiguidade na interpretação do modelo.
- **CPF override no código, nunca confiado ao modelo**: mesmo quando uma
  ferramenta aceita `cpf` como parâmetro, os agentes sempre usam o CPF
  autenticado guardado no estado da sessão, nunca o que o modelo eventual-
  mente "lembrar" — evita que uma alucinação do modelo vaze dados de outro
  cliente.
- **Streamlit** para a UI de testes, com CSS customizado que sobrescreve
  explicitamente os containers nativos do Streamlit (`data-testid`) para
  garantir que **todo** o app — não só o conteúdo principal — respeite o
  tema escolhido pelo botão próprio de claro/escuro.

---

## 🚀 Tutorial de Execução e Testes
Site utilizado para pegar a API-key
https://console.groq.com/
estou enviando o projeto ja com uma key no .env , caso deseje 
alterar,fique avontade para criar uma propia
### 1. Instalação

```bash
git clone https://github.com/Coyoss/Banco-Agil.git
cd banco-agil
python -m venv .venv
.venv\Scripts\activate   
#Caso ocorra erro ao iniciar o .venv rode esse comando 
#(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
pip install -r requirements.txt
```

### 2. Executando a aplicação

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

### 3. Testando o fluxo completo

Clientes de exemplo já incluídos em `data/clientes.csv` (use qualquer
formato de digitação para CPF e data eles também aparecem dentro do
próprio app, no painel **"🧪 Dados de teste"** da barra lateral):

| Cliente | CPF | Data de nascimento | Score | Limite atual |
|---|---|---|---|---|
| Ana Beatriz | 104.332.181-00 | 12/04/1990 | 620 | R$ 3.000,00 |
| João Pedro | 026.542.351-14 | 20/11/1975 | 180 | R$ 1.000,00 |
| Fernanda Costa | 083.863.794-99 | 05/01/1998 | 740 | R$ 7.000,00 |

Roteiro sugerido de teste:
1. Informe um CPF/data válidos → confirme a autenticação.
2. Peça para consultar o limite de crédito e, em seguida, o score.
3. Peça um aumento de limite → confirme quando perguntado → veja o
   resultado. Tente de novo e **recuse** a confirmação para ver que nada é
   registrado no CSV.
4. Peça um aumento alto o suficiente para ser **rejeitado** (ex.: cliente
   com score 180 pedindo R$ 5.000, confirmando o pedido) → aceite a
   entrevista de crédito oferecida → responda as 5 perguntas → veja a
   reavaliação automática do pedido.
5. No meio de uma conversa sobre crédito, pergunte a cotação de uma moeda
   (e vice-versa) para testar a troca de assunto implícita.
6. Peça para encerrar o atendimento a qualquer momento.
7. Teste também 3 tentativas seguidas de autenticação com dados incorretos
   para ver o encerramento automático por segurança.
8. Durante o processamento de qualquer mensagem, note que o campo de texto
   fica desabilitado até a resposta do agente aparecer.

---

## 📦 Tecnologias Utilizadas

- [LangGraph](https://github.com/langchain-ai/langgraph) — orquestração multiagente
- [LangChain](https://github.com/langchain-ai/langchain) + [langchain-groq](https://github.com/langchain-ai/langchain-groq) — *tool calling* e integração com o LLM
- [Groq API](https://console.groq.com/) (openai/gpt-oss-120b) — modelo de linguagem
- [Streamlit](https://streamlit.io/) — interface de testes
- [pandas](https://pandas.pydata.org/) — leitura/escrita dos CSVs
- [python-dateutil](https://dateutil.readthedocs.io/) — parsing flexível de datas
- [open.er-api.com](https://www.exchangerate-api.com/) — cotação de câmbio em tempo real (acesso aberto, sem chave)
