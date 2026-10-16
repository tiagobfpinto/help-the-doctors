# Help the doctor

Pergunta: **How does cross-label duplication affect the evaluation and prediction
of medical specialties from clinical text?** O percurso atual é
`03_experiencia_controlada.ipynb`. Comparamos NB, regressão logística (LR) e
LinearSVC, com e sem exclusão de classes observadas para o mesmo input.
LR/SVC comparam originais, recombinação de secções e cópias de controlo.

Cinco folds por **linhas** admitem notas conhecidas; cinco folds agrupados por
**nota** avaliam notas inéditas. São cenários separados e desenvolvimento
exploratório, sem teste independente após seleção. Os scores oficiais são
desconhecidos.

## Reproduzir

Na raiz, com Python 3.14.3:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-submission.txt
.\.venv\Scripts\python.exe classify.py
```

O último comando verifica `selected_model.json`, dados e código, parte dos
**2.669 originais** e gera as misturas da configuração selecionada antes de
ajustar o modelo. Produz **398 previsões** sem cabeçalho,
na ordem do teste limpo. Dispensa modelos guardados e outputs de validação.
O teste só é aberto depois de fixados o modelo e o índice de treino.

Para reproduzir também a experiência, seleção e paper:

```powershell
.\.venv\Scripts\python.exe run_final_evaluation.py
.\.venv\Scripts\python.exe classify.py --select
.\.venv\Scripts\python.exe build_paper_assets.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_paper.ps1 -Offline
```

O runner exige plano congelado e cinco exemplos inspecionados; recusa fontes
alteradas ou sobrepor uma execução existente. Antes de repetir, preservar a
pasta de resultados atual noutro local. Os manifestos registam versões, fontes,
folds e hashes; `--select` verifica a experiência antes de aplicar a regra.

As dependências de Jupyter estão em `requirements.txt`. O notebook reutiliza
uma experiência concluída. Os notebooks 01/02 e os scripts `audit_duplicate_*`
documentam exploração anterior; não são confirmação independente.

## Modelos e exclusão

NB segue a especificação da baseline: TF-IDF de `description` em minúsculas,
Porter stemming, alpha=1 e priors empíricos. LR/SVC usam blocos TF-IDF separados
de `sample_name` e `transcription`, palavras/bigramas, min_df=2, TF sublinear,
C=0.25, pesos equilibrados dos originais e até 2.000 iterações.

A exclusão usa o par **sample_name + transcription**, em minúsculas e com
espaços normalizados. O índice contém só originais do treino do fold. Num input
conhecido, exclui todas as classes nas suas cópias e escolhe o maior score
restante. Sem correspondência, com input totalmente vazio ou todas as classes
bloqueadas, mantém a previsão normal. NB também usa este par no índice,
embora o classificador leia apenas a descrição.

A hipótese é que outra anotação do mesmo input tende a usar uma classe ainda
não observada. É uma regularidade deste corpus, sem garantia clínica ou de
transferência. A avaliação agrupada verifica que a exclusão fica inativa.

## Augmentation e promoção

Nas quatro classes com menos de 100 originais, uma variante substitui uma
secção por outra com o mesmo heading, de outro original da mesma classe e de
outro grupo de duplicados. Ambos pertencem ao treino do mesmo fold. Notas sem
secções elegíveis são ignoradas; título e outras secções são conservados.

Estas misturas **não têm coerência clínica validada**: podem combinar idades,
sexos e diagnósticos incompatíveis. Cinco exemplos fixos são inspecionados
antes dos ajustes. Cópias dos mesmos originais, no mesmo número e com os mesmos
pesos, controlam a repetição. TF-IDF é reajustado por condição.

`RECOMBINATION_PLAN.md` e `augmentation_decision_plan.json` registam a regra
anterior aos novos scores: uma mistura só substitui o SVC com exclusão se
melhorar **macro F1 e F1 de Dermatology** face ao original da sua família **e ao
SVC atual**, perdendo no máximo um ponto de accuracy e mantendo accuracy >49%
e todos os F1 >=25%, no cenário por linhas com exclusão. Entre candidatas
admitidas, priorizam-se macro F1, Dermatology F1 e accuracy. Se nenhuma passar,
mantém-se o SVC. Cópias e scores agrupados são diagnósticos.

## Dados, paper e entrega

A preparação repara a estrutura CSV conservando rótulos, ordem e texto
disponível. Conservam-se casos incompletos e anotações em conflito.
`keywords` é excluído: os 2.219 valores não vazios começam pela classe.
`train.csv` e o teste limpo recebido são preservados byte a byte. O leitor
do teste exige quatro campos e 398 registos sem cabeçalho.

`relatorio.md` é o texto de trabalho; `paper.tex` segue o template:
duas colunas, 10 pt e Bera. As referências compactas impressas estão em
`paper_assets/references.tex`, com os metadados completos em `paper.bib`.
Tectonic está em `.tools/tectonic/`; num checkout novo, executar
`install_latex.ps1`. O primeiro build pode
precisar de rede, omitindo `-Offline`. O build regista hashes em
`output/pdf/compilation.json` e rejeita mais de três páginas no PDF completo,
incluindo a bibliografia. O enunciado/template exclui formalmente a bibliografia,
mas a preferência mais recente do utilizador impõe este limite mais estrito de
três páginas totais.

A identidade está preenchida para o **grupo 64**: Tiago Pinto, Gonçalo Narciso
Rolo, Gabriel Alexandre Lino Simões Ferreira e Miguel Neves Antão Ermida Parreira,
com **25% de contribuição cada**. O sistema selecionado é SVC com recombinação
e exclusão; a reprodução das previsões permanece inalterada e foi verificada
numa pasta limpa com resultados idênticos.

Para recompilar o paper e voltar a gerar os ficheiros finais do grupo 64,
**na raiz deste projeto completo**, executar:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_paper.ps1 -Offline
.\.venv\Scripts\python.exe prepare_submission.py --group 64
```

Estes comandos voltam a verificar o limite de três páginas totais, incluindo
bibliografia, e geram `output/submission/64.pdf` e `64.zip`, usando as
previsões já verificadas. Não é necessário repetir a avaliação, a seleção,
a inferência ou a geração das tabelas. Os fontes do paper e as ferramentas
de compilação/empacotamento permanecem neste projeto; o ZIP curado contém
os ficheiros de entrega e de reprodução das previsões.

O enunciado exige **NUM.zip** com **NUM.pdf**, **results.txt** e **código**,
sem modelos treinados. Para este grupo, `prepare_submission.py --group 64`
cria o pacote final; `--draft` gera apenas um rascunho de verificação.
Incluem-se os CSVs originais, dependências,
plano e exemplos para reprodução numa pasta limpa; excluem-se ambientes e caches.

Entrega pelo Fénix até **16 de outubro de 2026, às 23h59**. Os quatro pontos
automáticos exigem accuracy **>49%** (2 pontos) e, adicionalmente, F1 **>=25%
em todas as classes** (mais 2 pontos), medidos no teste oficial.

| Caminho | Conteúdo |
|---|---|
| outputs/avaliacao_recombinacao/ | Experiência atual, OOF, métricas, erros e decisão |
| selected_model.json | Sistema, dois cenários e proveniência |
| outputs/final/ | Inferência e diagnóstico de exclusão, sem scores oficiais |
| paper_assets/ | Tabelas e matriz verificadas |
| output/pdf/ | PDF e hashes das fontes |
| output/submission/ | Pacote curado e manifesto |
| outputs/before_section_recombination/ | Snapshot das fontes e entrega anteriores |
| outputs/avaliacao_final/ | Experiência histórica com remoção de artigos |

Verificação:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
