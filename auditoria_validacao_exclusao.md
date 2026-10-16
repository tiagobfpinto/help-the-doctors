**Auditoria da validação e da exclusão de classes em notas repetidas — 16 de outubro de 2026**

**Atualização após a auditoria:** a correção está integrada em `duplicate_exclusion.py`,
`run_final_evaluation.py` e `classify.py`. A comparação final avalia os 14 candidatos
nos dois protocolos; a seleção está em `selected_model.json`, e as 398 previsões
foram regeneradas. As secções seguintes preservam a sequência e os resultados da
auditoria exploratória que motivou a mudança. O percurso atual está em `README.md`.

Esta auditoria corrige a recomendação anterior de fechar a entrega com o SVC sem pós-processamento, baseada sobretudo nos 57,10% de accuracy da validação agrupada. Esse resultado é reproduzível, mas responde ao cenário de notas novas. Faltava avaliar o comportamento perante novas linhas cujas notas já aparecem no treino com outras especialidades. A confiança anterior no desempenho da submissão foi excessiva.

As experiências desta auditoria usam exclusivamente os 2.669 casos de `train.csv`. Não leem o teste oficial, não utilizam rótulos de teste, nem substituem `results.txt` ou a seleção final. Os números do teste citados na revisão externa não são usados para escolher a regra ou o protocolo. As métricas seguintes são desenvolvimento exploratório, não avaliação oficial nem avaliação independente após seleção.

**O padrão observado no treino**

- Há 731 grupos de transcrições não vazias repetidas, envolvendo 1.530 linhas: 665 pares, 64 trios e dois grupos de quatro.
- Em 730 grupos, envolvendo 1.528 linhas, as classes das cópias são todas diferentes. Nesses grupos, descrição e título também coincidem.
- O único grupo que repete a mesma classe contém os casos 1295 e 1377, ambos Orthopedic. A transcrição é apenas `CHIEF COMPLAINT:`; os títulos são diferentes, Shoulder Contusion e Hip Pain. Igualdade de uma transcrição curta não garante identidade do caso clínico.
- Com `StratifiedKFold(5, shuffle=True, random_state=0)`, 1.261 casos de validação têm uma transcrição não vazia presente no treino do respetivo fold. Em 1.259 deles (99,84%), a classe verdadeira está ausente das classes dessas cópias. As duas exceções são o par anterior.
- Ao exigir igualdade do par título/transcrição e pelo menos um campo não vazio, há 1.269 correspondências, todas com classe verdadeira ausente das cópias: 1.259 transcrições não vazias e dez vazias com título igual.

Estes factos são verificáveis em `outputs/auditoria_exclusao/dataset_structure.json` e no script que o acompanha. A unicidade observada de `(input, classe)` explica a complementaridade em divisões deste corpus finito. Não demonstra uma regra clínica nem garante o mesmo mecanismo no teste ou noutra coleção.

**Reprodução da experiência indicada pelo Claude**

Mantivemos campos, hiperparâmetros e pesos dos modelos anteriores. Todos os modelos usam os mesmos cinco folds estratificados por linhas, com seed 0. Vocabulário, IDF, pesos e índice de cópias são ajustados exclusivamente no treino de cada fold.

A regra primária reproduzida compara transcrições não vazias, normalizadas em minúsculas e espaços. Exclui todas as classes observadas para essa transcrição no treino e escolhe a classe restante com maior score. Usa `decision_function` em LR/SVC e log-probabilidades em NB. Se não houver correspondência, conserva a previsão; se todas as classes forem excluídas, conserva a previsão original e regista o fallback. A função de decisão não recebe rótulos de validação.

| Sistema, OOF por linhas | Accuracy (%) | Macro F1 (%) | F1 mínimo (%) |
|---|---:|---:|---:|
| NB | 38,67 | 13,23 | 0,00 |
| NB + exclusão | 55,26 | 35,14 | 0,00 |
| LR balanced | 48,78 | 51,81 | 31,86 |
| LR balanced + exclusão | 76,36 | 73,09 | 47,76 |
| SVC balanced | 46,16 | 49,77 | 23,82 |
| SVC balanced + exclusão | 77,37 | 75,13 | 48,15 |

Os resultados reproduzem os valores arredondados da revisão externa. No SVC, a regra altera 962 previsões: corrige 835, estraga duas e mantém 125 erradas, produzindo mais 833 acertos líquidos. Nos 1.261 casos com cópia, a accuracy passa de 22,20% para 88,26%. Nos 1.408 casos sem cópia, mantém-se exatamente em 67,61%. As duas previsões prejudicadas são as transcrições boilerplate acima.

Repetindo a divisão com seeds 1 e 2, o SVC sem regra obtém 46,27% e 46,38%; com a regra, 77,74% e 77,52%. A LR com regra obtém 76,21% e 76,47%. Na avaliação agrupada, a regra não encontra cópias nem altera previsões: reproduz exatamente NB 41,14%, LR 55,75% e SVC 57,10%. Este controlo confirma que o ganho depende da disponibilidade de cópias no treino, não de uma melhoria em notas inéditas.

A sensibilidade posterior que exige igualdade dos inputs completos (`sample_name`, `transcription`), ignorando apenas inputs totalmente vazios, conserva os mesmos modelos e scores OOF. O SVC obtém 77,56%, 78,01% e 77,71% de accuracy nas seeds 0/1/2; os F1 mínimos são 48,15%, 57,14% e 53,85%. A LR obtém 76,62%, 76,58% e 76,81%. Não há danos observados nesta variante, e os dois casos de boilerplate deixam de coincidir. A variante admite correspondências por título quando falta transcrição; esses casos são contabilizados separadamente. Trata-se de uma sensibilidade motivada após inspeção, não de confirmação independente nem de evidência adicional do teste oficial.

**Interpretação e limites**

A validação agrupada mantém a mesma nota integralmente num lado da divisão e estima generalização a notas não vistas. A validação por linhas admite notas conhecidas e estima outra situação de amostragem. Neste corpus, as cópias têm sobretudo rótulos concorrentes: a sua presença não implica necessariamente otimismo da validação por linhas. Não se deve escolher o protocolo que produz o maior score. A diferença entre os cenários deve ser apresentada explicitamente. Ver a [documentação oficial de validação por grupos](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

A exclusão explora uma hipótese específica: a próxima linha de uma nota tende a corresponder a uma especialidade ainda não observada nas suas cópias. É um classificador condicionado pela construção do corpus. Não resolve a ambiguidade médica nem prova melhor compreensão clínica. Repetir seeds verifica sensibilidade à partição dos mesmos dados; não valida essa hipótese numa coleção independente.

Comparar um input, durante inferência, com textos e rótulos disponíveis no treino não requer rótulos de teste. Já usar estatísticas do teste para selecionar regras ou protocolos seria uma utilização adicional desse conjunto, incompatível com a separação estabelecida para este projeto. Ver também a [documentação oficial sobre data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Por isso, esta auditoria não confirma os 239 matches do teste, não calcula métricas nesse teste e não trata a estimativa externa de 40% como um resultado observado.

**Consequências para a entrega**

A recomendação de manter o SVC atual deve ser revista. Uma integração da exclusão deve exigir correspondência rigorosa dos inputs, registar correspondências e fallbacks, e usar o mesmo procedimento na validação e inferência final. A variante por título/transcrição é uma sensibilidade separada, motivada pelo contraexemplo de boilerplate; os seus números não devem ser confundidos com a reprodução primária acima.

Não basta acrescentar um parágrafo ao paper se o sistema final mudar: a seleção, os resultados por classe, a matriz e a narrativa devem identificar corretamente o sistema e o cenário avaliados. Os três erros agrupados existentes continuam válidos se forem explicitamente apresentados como erros no cenário de notas novas, onde a regra fica inativa; não podem ser descritos como erros de outro protocolo sem verificação. Os resultados agrupados anteriores mantêm valor. A augmentation testada também mantém o seu resultado negativo, limitado à remoção conservadora de artigos e ao respetivo protocolo. Não demonstra a ineficácia geral de augmentation clínica.

A pergunta poderá passar a centrar-se em como a estrutura dos rótulos de notas repetidas afeta classificação e avaliação, deixando weighting e augmentation como comparadores. O paper deve declarar a hipótese de complementaridade, o caráter específico do corpus, a seleção exploratória e a contribuição desta revisão externa do Claude, além da assistência do Codex.

A observação externa de que o PDF não foi compilado está desatualizada: existe `output/pdf/paper.pdf`, com a compilação e a contagem de três páginas de corpo registadas em `output/pdf/compilation.json`. Os dados do grupo permanecem por preencher por decisão do utilizador.

**Reprodução e artefactos**

```powershell
.\.venv\Scripts\python.exe audit_duplicate_exclusion.py
.\.venv\Scripts\python.exe audit_duplicate_key_sensitivity.py
.\.venv\Scripts\python.exe outputs/auditoria_exclusao/dataset_structure.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`outputs/auditoria_exclusao/plan.json` regista o desenho antes dos ajustes. `summary.csv`, `metrics_per_fold.csv`, `metrics_by_subgroup.csv`, `metrics_per_class.csv`, `oof_predictions.csv` e `rule_diagnostics.csv` permitem conferir todos os valores e intervenções. `experiment.json` regista hashes, versões e a conclusão da execução. As variantes de correspondência, quando avaliadas, ficam em artefactos de sensibilidade distintos.

A execução exploratória acima terminou em 206 segundos, com 60 ajustes, zero warnings e todos os hashes de origem inalterados. Nessa etapa, a suite tinha 47 testes aprovados e os ficheiros de entrega ainda não tinham sido alterados. A integração posterior usa o par completo de inputs em validação e inferência, com 70 ajustes adicionais e manifesto próprio em `outputs/avaliacao_final/`. A versão anterior das previsões e seleção foi preservada em `outputs/final/before_exclusion/`; treino e teste originais permanecem intactos.
