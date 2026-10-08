# Help the doctor

O notebook `01_preparacao_dados.ipynb` executa a reparação estrutural e a auditoria
do `train.csv`. As funções estão em `data_preparation.py`, para poderem também ser
testadas e reutilizadas. O dataset original e `test_no_labels.csv` não são alterados.

O notebook **`02_treino_avaliacao.ipynb`** contém o percurso completo: preparação
dos dados originais, exploração das classes, pré-processamento do texto, TF-IDF,
treino e validação cruzada, comparação com baselines, accuracy, precision, recall
e F1 globais e por classe, matrizes de confusão e análise dos erros. No fim, treina
o modelo final, guarda as métricas e o modelo e gera as previsões do teste.
Pode ser executado diretamente, sem correr primeiro o notebook 01.

## Executar

Para gerar diretamente o dataset, os relatórios e o leitor, na pasta do projeto:

```powershell
python data_preparation.py
```

Esta alternativa usa apenas a biblioteca padrão de Python e as mesmas funções do
notebook. Os caminhos são calculados a partir da localização do script.

Para explorar também as tabelas e explicações no notebook, cria um ambiente Python
e instala as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Abre o notebook e executa **Restart Kernel and Run All Cells**. Em VS Code, seleciona
o Python de `.venv` como kernel e usa **Run All**.

No notebook 02, altera apenas a célula **Configuração da experiência** para escolher
os campos, o modelo principal, os folds e os parâmetros. O modelo inicial é uma
regressão logística com TF-IDF separado por campo e pesos de classe equilibrados;
é comparado com o Naive Bayes da descrição e com a classe mais frequente. `keywords`
fica excluído porque contém a especialidade. A validação estratificada reproduz
o protocolo dos scripts; `CV_MODE = "grouped"` mantém transcrições repetidas no mesmo
fold para avaliar a generalização a outras notas. O notebook mostra a sobreposição
de transcrições em cada fold e identifica o protocolo nas métricas exportadas.

Na secção **9.1**, `COMPARE_LR_FIELDS = True` compara os três campos individualmente,
os três pares e todos juntos, mantendo a regressão logística, os parâmetros TF-IDF
e os folds fixos. A tabela mostra accuracy, precision/recall/F1 macro, F1 mínimo
e os critérios do projeto. Esta etapa acrescenta alguns minutos à execução.
As tabelas por classe e fold, as previsões OOF, os parâmetros e o gráfico ficam em
`outputs/comparacao_campos_lr/`. Usa `COMPARE_LR_FIELDS = False` para omitir esta etapa.

Os resultados desta experiência ficam em `outputs/notebook/`: tabelas CSV,
gráficos, previsões de validação, `experiencia.json`, `modelo.joblib` e `results.txt`
(409 previsões, sem cabeçalho). O `results.txt` na raiz só é substituído se
`WRITE_ROOT_RESULTS = True`. O teste não tem rótulos; as métricas são calculadas
apenas nas previsões de validação cruzada do treino.

Instalação do Jupyter: https://jupyter.org/install

## Regras de reparação

- Uma descrição com uma única aspa de abertura e sem os restantes campos é
  preservada. Os campos indisponíveis ficam vazios e o caso é marcado como parcial.
- Dois campos vazios excedentes são removidos apenas quando existem sete campos
  e os primeiros cinco têm a estrutura esperada.
- Uma continuação sem especialidade é ligada à transcrição do caso imediatamente
  anterior apenas quando esse caso tinha campos vazios excedentes. É inserida uma
  quebra de linha. Caracteres em falta não são reconstruídos e aspas que já fazem
  parte do texto descodificado são preservadas.
- Estruturas desconhecidas interrompem a execução. Esta reparação por linha física
  aplica-se ao ficheiro fornecido, não a CSVs arbitrários com campos multilinha.

Os rótulos, a ordem dos casos e todo o texto disponível são preservados. Não se
eliminam classes, textos repetidos nem casos com campos vazios. No relatório, uma
transcrição repetida com rótulos diferentes é uma ambiguidade assinalada, não uma
decisão automática de correção. Uma transcrição vazia não é agrupada com outros vazios.

## Conjunto de teste

```powershell
python read_test_set.py
```

`read_test_cases()` é a leitura que o `predict.py` deve usar, e `write_results()`
escreve o `results.txt`, uma especialidade por linha, sem cabeçalho. A linha N do
`results.txt` corresponde ao registo CSV N, não à linha física N.

- O `test_no_labels.csv` não tem cabeçalho. `pd.read_csv(..., sep=";")` toma o
  primeiro caso como cabeçalho e devolve 408 casos em vez de 409.
- O ficheiro tem 452 linhas físicas e 409 registos CSV. Lido e reescrito com o
  módulo `csv`, reproduz o original byte a byte, por isso é a exportação de uma
  tabela com 409 casos.
- Quatro registos têm uma descrição que se estende por 9 a 15 linhas físicas. As
  linhas extra começam por uma especialidade: são texto de outros casos que ficou
  dentro da célula, e não casos do teste. O modelo recebe apenas a primeira linha.
  Os rótulos embutidos ficam na auditoria e não são usados.
- Cada um desses registos é seguido por uma linha só com keywords. No treino
  seria ligada ao caso anterior. Aqui é um registo da tabela e recebe uma previsão.
- Nenhum registo é removido ou juntado. O número esperado (409) está fixo em
  `EXPECTED_TEST_CASES`, para que uma alteração na leitura pare a execução.

## Modelos

```powershell
.\.venv\Scripts\python.exe classify.py        # CV no treino e depois results.txt
.\.venv\Scripts\python.exe explore_tfidf.py   # heatmaps de TF e TF-IDF
.\.venv\Scripts\python.exe compare_logistic_fields.py  # os 7 conjuntos de campos, mesma LR e folds
```

| Ficheiro | Responsabilidade |
|---|---|
| `config.py` | Caminhos e parâmetros: campos de texto, folds, seed, critérios |
| `data.py` | Carrega treino e teste a partir dos ficheiros originais, com a reparação |
| `models.py` | Naive Bayes e regressão logística com TF-IDF por campo, em Pipelines |
| `evaluation.py` | Validação cruzada estratificada e verificação dos critérios |
| `classify.py` | Avalia o modelo escolhido e escreve o `results.txt` |
| `explore_tfidf.py` | Exploração das matrizes TF e TF-IDF |
| `compare_logistic_fields.py` | Comparação dos campos com a mesma regressão logística e os mesmos folds |

## Ficheiros gerados

| Ficheiro | Conteúdo |
|---|---|
| `data/processed/train_clean.csv` | Dataset reparado, com os cinco campos originais |
| `outputs/auditoria/repairs.csv` | Cada alteração, a linha original e a regra aplicada |
| `outputs/auditoria/record_audit.csv` | Correspondência entre registos e linhas originais, campos vazios e grupos repetidos |
| `outputs/auditoria/summary.json` | Contagens e hashes dos ficheiros de origem e reparado |
| `outputs/dados-legiveis/ler_dataset_clean.html` | Leitor do dataset reparado, com pesquisa, filtros e anotações |
| `data/processed/test_clean.csv` | Teste lido, um caso por registo, pela ordem do `results.txt` |
| `outputs/auditoria/test_record_audit.csv` | Linhas de origem de cada caso de teste, reparações e rótulos embutidos |
| `outputs/auditoria/test_summary.json` | Contagens do teste, incluindo a comparação com o pandas |

Os CSVs usam `;` e UTF-8 com BOM. Os ficheiros gerados são ignorados pelo Git e
podem ser recriados executando o script ou o notebook. O leitor e o Excel criados anteriormente
referem-se à interpretação inicial do CSV; o novo leitor tem números de registo
diferentes e conserva a correspondência com as linhas do ficheiro original.

## Verificações

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O notebook verifica também a cobertura de todas as linhas de origem, a preservação
da ordem e das especialidades, e a igualdade dos campos após exportar e reler o CSV.
Uma nova execução substitui apenas os ficheiros gerados pelo mesmo processo.

Para avaliar a generalização a notas novas, os splits devem manter cada grupo de
transcrições repetidas no mesmo conjunto (`CV_MODE = "grouped"` no notebook 02).
O protocolo estratificado dos scripts permite repetições entre folds; os seus
resultados respondem a uma condição de avaliação diferente. TF-IDF e eventual
data augmentation devem ser ajustados apenas ao treino de cada fold.
