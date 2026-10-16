# Reproduzir a comparação controlada e a análise dos erros

**Nota histórica:** este documento descreve a experiência agrupada anterior.
A avaliação e a seleção finais, com os dois protocolos e o decoder de exclusão,
são reproduzidas por `run_final_evaluation.py`, `classify.py --select` e pelo
notebook 03. Ver `README.md`. Os números abaixo continuam válidos para o desenho
histórico identificado, mas não definem a seleção final atual.

Executar a partir da pasta do projeto, com o ambiente em `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_augmentation -v
.\.venv\Scripts\python.exe run_augmentation.py
.\.venv\Scripts\python.exe explain_errors.py
```

Os scripts leem apenas `train.csv`. Não leem o teste sem rótulos, não escrevem `results.txt` e não substituem o modelo final. A comparação demora cerca de três minutos no ambiente registado; a análise dos erros reajusta apenas três folds do SVC. Os resultados ficam em `outputs/augmentation_controlada/`, que está excluída do Git e pode ser recriada.

O desenho foi fixado em `design.json` antes do primeiro ajuste: cinco folds agrupados, seed 0, TF-IDF separado de `sample_name` e `transcription`, unigramas/bigramas, `min_df=2`, frequência sublinear e `C=0.25`. O agrupamento usa transcrição normalizada; quando esta falta, usa o título. Sete entradas totalmente vazias recebem grupos individuais e continuam a ser avaliadas. As previsões OOF incluem todos os 2 669 originais. Este é um conjunto de desenvolvimento anteriormente explorado; a seleção do melhor resultado OOF não é uma avaliação independente.

As condições A–E são regressões logísticas: A sem pesos/sem augmentation; B balanced/sem augmentation; C sem pesos/article dropout; D balanced/article dropout; E balanced/cópias das mesmas fontes aceites em C/D. NB é o baseline de descrição com Porter stemming; SVC é LinearSVC balanced, sem augmentation. A escolha final segue a regra registada: entre configurações com F1 mínimo de pelo menos 0,25, maior accuracy, depois macro F1 e F1 mínimo. Isso selecionou SVC.

O dropout altera apenas `transcription`: remove `max(1, floor(0.20*m))` das `m` ocorrências isoladas de `the`, de forma determinística por seed/fold/ID. Não significa remover 20% de todas as palavras. As quatro classes elegíveis foram fixadas pelos suportes originais inferiores a 100; não se filtram conflitos de anotação. Foram aceites 864 das 872 visitas de fontes elegíveis aos folds de treino; removeram-se 5 964 de 482 888 palavras dessas fontes (1,235%). D e E diferem em apenas três previsões OOF. Este resultado não permite generalizar para outras estratégias de augmentation.

Os pesos balanced são calculados apenas dos originais de cada fold, fixados antes de acrescentar filhos/cópias e normalizados para somar o número de originais. D/E têm as mesmas fontes, contagens e pesos; ambos também refazem TF-IDF com os exemplos acrescentados. Isso controla a reponderação das fontes sem atribuir automaticamente um resultado a nova informação lexical.

**Nota sobre nomes nos metadados de pesos:** `weights_per_fold.json` e os metadados do ajuste final reutilizam o lote A para NB e B para SVC. Por isso, o campo `name` do lote pode conter “LR” e `condition` pode ser A/B, mesmo quando `model_id` é NB/SVC. Esses campos identificam o lote de ponderação reutilizado; não identificam o estimador treinado. O estimador real é o indicado por `model_id`, `summary.csv`, `design.json` e `build_evaluated_model()`: NB com stemming e SVC com a configuração acima. Não há dupla aplicação de class weights.

`explain_errors.py` reproduz os folds 3, 4 e 5 necessários aos casos 722, 298 e 308. Confirma todas as previsões dos respetivos folds contra OOF e decompõe a diferença de decision scores do SVC em `x_j * (coef_prevista_j - coef_rótulo_j)` e diferença de intercept separada. As margens foram reconstruídas com erro absoluto inferior a 5e-16. O CSV contém todas as contribuições ativas, com ordenação para consultar as maiores; os metadados incluem contagens de documentos por termo e campo. É uma explicação local das associações aprendidas, não uma explicação causal ou uma validação clínica.

O código de reprodução é preservado na raiz mesmo quando os outputs ignorados pelo Git não estão presentes. A lógica partilhada de preparação/treino está em `augmentation.py`; não se devem trocar parâmetros no script de erros sem repetir e identificar a experiência correspondente.
