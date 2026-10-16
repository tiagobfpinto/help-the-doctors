# Help the doctor

O percurso atual está em **03_experiencia_controlada.ipynb**. A pergunta é:
**How does cross-label duplication affect the evaluation and prediction of
medical specialties from clinical text?** O projeto compara Naive Bayes,
regressão logística e LinearSVC, pesos de classe, augmentation conservadora,
cópias de controlo e exclusão das classes já observadas para o mesmo input.

A avaliação final mantém dois cenários explícitos: cinco folds estratificados
por **linhas** e cinco folds agrupados por **nota**. As mesmas configurações
são avaliadas nos dois, sem escolher o protocolo com o maior score. A seleção
usa desenvolvimento por linhas; o resultado agrupado do mesmo candidato
descreve generalização a notas inéditas. Ambos são desenvolvimento exploratório
após inspeção deste corpus, sem avaliação independente após seleção.

## Executar e reproduzir

Na pasta do projeto, criar um ambiente Python e instalar dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O percurso completo, incluindo a auditoria e as sensibilidades de seeds 1/2:

```powershell
.\.venv\Scripts\python.exe audit_duplicate_exclusion.py
.\.venv\Scripts\python.exe audit_duplicate_key_sensitivity.py
.\.venv\Scripts\python.exe run_final_evaluation.py
.\.venv\Scripts\python.exe classify.py --select
.\.venv\Scripts\python.exe build_paper_assets.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_paper.ps1 -Offline
```

Os dois primeiros comandos reproduzem a revisão exploratória. A avaliação
final refaz a comparação principal com o helper de produção; reutiliza
opcionalmente os scores das seeds 1/2 para verificar a equivalência do decoder,
sem os usar na seleção. Sem aquela auditoria, a comparação principal continua
completa, mas a sensibilidade adicional não estará disponível.

Apenas o quarto comando lê o teste oficial: primeiro seleciona o candidato e
ajusta o classificador e o índice em todos os **2.669 originais**; depois gera
as **398 previsões**, uma especialidade por linha, sem cabeçalho e na ordem
original. Nenhuma estatística de correspondências do teste entra na escolha
do modelo. O teste não tem rótulos e não tem métricas medidas.

Para regenerar apenas as previsões, usar classify.py sem --select. Esse
comando reutiliza selected_model.json e verifica hashes do treino, código e
configuração. Alterações de dados, implementação ou avaliação requerem repetir
as etapas correspondentes; as verificações rejeitam seleções antigas ou
artefactos incompatíveis em vez de reutilizá-los silenciosamente.

O notebook 03 permite explorar as mesmas etapas. Por omissão reutiliza uma
avaliação concluída e verificada; REPEAT_EXPERIMENT=True volta a executá-la.
Os notebooks 01 e 02 documentam preparação e exploração anteriores; o 02
é histórico e não substitui a seleção final atual.

## Classificação e exclusão

NB segue a baseline especificada: TF-IDF de description em minúsculas,
Porter stemming, MultinomialNB com alpha=1 e priors empíricos. LR e SVC usam
TF-IDF separado de sample_name e transcription, unigramas/bigramas,
min_df=2, TF sublinear, C=0.25 e até 2.000 iterações. A implementação e as
versões efetivamente usadas ficam registadas nos manifestos.

A chave da exclusão é o par completo **sample_name + transcription**, em
minúsculas e com espaços normalizados. O índice contém só os originais do
treino do fold, nunca validação nem descendentes sintéticos. Inputs
totalmente vazios não coincidem. Para um input conhecido, o decoder exclui
todas as classes presentes nas suas cópias de treino e escolhe o maior score
restante. Sem correspondência, ou se todas as classes forem excluídas,
mantém a previsão original. NB também usa esse par no lookup, embora o seu
classificador leia apenas description.

A regra explora uma regularidade da anotação: outra cópia tende a representar
uma especialidade ainda não observada. **Não é uma regra médica nem uma
garantia para outra coleção.** A identidade completa evita o contraexemplo de
transcrições limitadas a CHIEF COMPLAINT: sob títulos diferentes. A validação
agrupada verifica que o decoder fica inativo em notas inéditas.

Cada uma das sete configurações base é avaliada com decoder normal e com
exclusão, totalizando 14 candidatos em cada cenário. Entre candidatos com F1
mínimo >=25%, seleciona-se a maior accuracy por linhas, depois macro F1 e F1
mínimo, com desempate determinístico. Se nenhum cumprir a restrição, prioriza-se
macro F1. O requisito de accuracy é estritamente >49%. A escolha não constitui
uma medição dos quatro pontos automáticos no teste oficial.

## Augmentation e dados

Os braços LR são A: uniformes; B: balanced; C: uniformes + article dropout;
D: balanced + dropout; E: balanced + cópias das mesmas fontes aceites em D.
A transformação remove max(1,floor(0.2*m)) de m ocorrências isoladas de
the apenas da transcrição, numa variante por fonte elegível das quatro
classes com menos de 100 exemplos. Títulos e outros tokens são preservados.
A taxa é sobre ocorrências do artigo, não 20% do texto. Não há validação
clínica nem geração de casos por LLM.

Os pesos de classe são estimados nos originais de cada treino; a massa total
é normalizada ao número de originais. D/E têm as mesmas fontes, contagens e
pesos, refazendo TF-IDF nos respetivos textos de treino. Validação e teste
nunca são aumentados. A conclusão limita-se a esta transformação e orçamento.

data_preparation.py repara a estrutura CSV conservando rótulos, ordem e texto
disponível. Campos incompletos e conflitos ficam documentados; não são
corrigidos por julgamento médico. keywords é excluído porque todos os seus
2.219 valores não vazios começam pela classe. train.csv e o teste limpo
recebido são preservados byte a byte. O leitor do teste usa csv sem cabeçalho,
exige quatro campos e 398 registos, sem as heurísticas da versão antiga.

## Paper e entrega

relatorio.md contém o texto de trabalho; paper.tex e paper.bib seguem o
template e usam os artefactos em paper_assets/. Tectonic está instalado
localmente em .tools/tectonic/. Num checkout novo, executar install_latex.ps1;
o primeiro build pode precisar de rede, omitindo -Offline.

O PDF fica em output/pdf/paper.pdf. O build regista hashes e contagem em
output/pdf/compilation.json e rejeita mais de três páginas de corpo.
A bibliografia está excluída desse limite pelo template. Grupo, autores e
percentagens de contribuição ficam por preencher, por decisão do utilizador:
preenchê-los e recompilar antes de entregar.

O enunciado exige **NUM.zip**, contendo **NUM.pdf**, **results.txt**,
**código do projeto** e eventuais datasets adicionais recolhidos. Exclui
os modelos treinados. Não incluir .venv/, .tools/, caches ou modelos joblib.
A submissão é pelo Fénix até **16 de outubro de 2026, às 23h59**.
Os datasets originais têm de estar disponíveis na raiz para reproduzir o código.

## Artefactos e verificações

| Caminho | Conteúdo |
|---|---|
| outputs/avaliacao_final/ | Desenho, hashes, 28 resultados, OOF, métricas por classe/fold, matrizes, erros e auditorias |
| selected_model.json | Candidato, classificador, decoder, dois cenários, regra de seleção e proveniência |
| outputs/final/inference.json | Ajuste completo, hashes e contagem do teste/previsões, sem métricas oficiais |
| outputs/final/decoding_diagnostics.csv | Decisões do decoder na inferência, pela ordem dos casos |
| outputs/final/before_exclusion/ | Cópia preservada das previsões e seleção anteriores à correção |
| paper_assets/ | Tabelas, matriz e métricas validadas para o paper |
| auditoria_validacao_exclusao.md | Motivação e reprodução da correção metodológica |
| outputs/auditoria_exclusao/ | Auditoria exploratória anterior, separada da comparação final |
| outputs/augmentation_controlada/ | Experiência agrupada histórica; não é a seleção final atual |

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os outputs ignorados pelo Git podem ser recriados. Os scripts e o notebook
não dependem de um modelo treinado previamente guardado.
