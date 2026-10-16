**Revisão científica do projeto e plano para um paper de três páginas**

**Nota posterior (16 de outubro):** esta é a revisão histórica de 15 de outubro.
A recomendação de seleção baseada no cenário agrupado foi revista após a análise
da complementaridade dos rótulos entre cópias. Os resultados agrupados continuam
válidos para notas inéditas; não são uma previsão do teste oficial. A correção,
com os dois protocolos e exclusão por input completo, está documentada em
`auditoria_validacao_exclusao.md` e no percurso atual de `README.md`.

Análise de 15 de outubro de 2026. Foram lidos o relatório, o enunciado, o template, os scripts de preparação/modelação/avaliação, os notebooks e os resultados guardados. A revisão inclui uma nova auditoria dos dados e pequenos ensaios diagnósticos, separados dos resultados anteriores. O teste oficial não foi usado para treinar, selecionar ou avaliar modelos nesta revisão. O `relatorio.md` e os ficheiros de entrega não foram alterados.

**A avaliação global é favorável, mas o trabalho ainda não está fechado como paper.** Há boa engenharia de dados, uma comparação de campos controlada e cuidado em distinguir resultados de hipóteses. O contributo científico atual é perceber como a representação e a ambiguidade das categorias afetam o desempenho. Falta executar a experiência que responde à pergunta sobre augmentation, integrar um terceiro modelo, apresentar resultados por classe e discutir erros concretos. A prioridade deve ser completar essa cadeia de evidência e condensar a escrita.

É importante separar três objetivos: ultrapassar a implementação local da baseline descrita pela professora; cumprir os limiares da avaliação automática; e escrever um paper cientificamente sólido. Já existe evidência para o primeiro. Os novos ensaios por grupos tornam o segundo plausível, mas não medem o teste oficial. O terceiro depende sobretudo das experiências controladas e da discussão, que valem muito mais na classificação do projeto do que perseguir pequenas melhorias num único score.

O enunciado marca a entrega para **16 de outubro de 2026, às 23h59**. Assim, o plano proposto privilegia experiências pequenas e uma pergunta que possa ser respondida com o trabalho disponível.

**O que o enunciado e o template exigem.** O [enunciado](Project-2026-Description.pdf), pp. 2–3, atribui 2 pontos a accuracy estritamente superior a 49% e outros 2 quando, adicionalmente, todas as classes têm F1 de pelo menos 25%. Estes limiares aplicam-se à avaliação automática do teste fornecido. Uma estimativa de validação cruzada acima deles não garante esses pontos. Os restantes 16 pontos são do paper.

O [template](Project-Template/Project-Template/template.tex) usa `article`, duas colunas, `10pt` e o pacote `bera`; o PDF fornecido mostra Bera Serif. O corpo tem no máximo três páginas. O template exclui a bibliografia desse limite e permite um apêndice de até uma página, apenas com figuras e tabelas complementares. O texto principal tem de ser compreensível sem esse apêndice. Se quiserem limitar o PDF inteiro a três páginas, é uma restrição adicional vossa: será preciso reservar também espaço para as referências. Não há motivo para trocar de fonte ou reduzir margens.

| Elemento da rubrica | Estado observado | Ação necessária |
|---|---|---|
| Pergunta explícita e narrativa, 1,5 pontos | Existe pergunta; ainda não há experiência que lhe responda | Restringir a formulação e alinhar comparação e discussão |
| Análise dos dados, 1,5 | Há observações concretas fortes | Comprimir; destacar desequilíbrio e ambiguidade separadamente |
| Data augmentation, 2 | Ainda é trabalho futuro | Executar e avaliar uma transformação textual real |
| Modelos, 4 | NB e LR no relatório; SVC agora testado nos pilotos | Integrar os três modelos e os detalhes necessários à reprodução |
| Protocolo, 1 | Pipelines e OOF bem definidos | Clarificar seleção de parâmetros e identidade das notas |
| Resultados, 1,5 | Métricas existentes; apresentação incompleta no texto | Tabela principal, resultados por classe e matriz completa |
| Discussão, 3 | Boa prudência; faltam exemplos concretos | Três erros com excerto, rótulo, previsão e interpretação |
| Trabalho futuro, 0,5 | Há uma lista de tarefas ainda obrigatórias | No paper final, reservar esta secção para extensões opcionais |
| Referências, 1 | Ausentes no relatório atual | Citar fontes e declarar o uso efetivo de LLMs |

As variantes dos campos são ablações da mesma LR. Não contaria com elas para cumprir, de forma segura, a exigência de pelo menos três modelos. NB, LR e LinearSVC são uma escolha curta e defensável. Um modelo pior também contribui se testar uma hipótese e for analisado.

**O estado real dos modelos.** Os resultados guardados da comparação de campos usam as mesmas cinco partições estratificadas, com shuffle e seed 0. Vocabulário, IDF e filtragem são aprendidos apenas no treino de cada fold. A LR usa TF-IDF separado por campo, unigramas e bigramas, frequência sublinear, `min_df=2`, `C=1` e pesos `balanced`.

| Configuração já existente, CV estratificada por linhas | Accuracy | Macro F1 | Menor F1 |
|---|---:|---:|---:|
| NB da descrição, lowercasing e Porter stemming | 38,67% | 13,23% | 0,00% |
| LR balanced, só descrição | 42,94% | 44,69% | 23,48% |
| LR balanced, três campos | 44,96% | 47,61% | 20,81% |
| LR balanced, título + transcrição | **46,38%** | **48,96%** | **25,45%** |

Fontes: [baselines.csv](outputs/comparacao_campos_lr/baselines.csv), [comparacao_campos.csv](outputs/comparacao_campos_lr/comparacao_campos.csv) e [experiencia.json](outputs/comparacao_campos_lr/experiencia.json). “Título” corresponde a `sample_name`.

A configuração de dois campos melhora a baseline local com stemming em **7,72 pontos percentuais de accuracy e 35,73 de macro F1**. Este resultado sustenta a configuração completa. Para isolar o algoritmo, a comparação mais informativa é NB e LR sobre a mesma descrição; aí a LR obtém mais 4,27 pontos de accuracy, embora representação e pesos ainda difiram. A ablação posterior isola os campos dentro de uma LR fixa. Esta distinção já está bem feita no relatório e deve manter-se numa frase curta.

A baseline local é uma implementação da especificação do enunciado, com Porter e os parâmetros documentados; não conhecemos todos os detalhes da implementação privada da professora. Além disso, 49% é um critério da avaliação, não uma medição da baseline nos vossos folds. No OOF atual, faltariam 70 acertos líquidos para passar de 1.238 para 1.308 acertos em 2.669 casos, ultrapassando 49%.

O F1 mínimo de 25,45% deixa apenas 0,45 pontos de margem. Os mínimos por fold da mesma configuração são aproximadamente 23,7%, 26,1%, 23,3%, 17,3% e 26,2%. O pooled OOF é válido como agregado escolhido, mas a margem sobre o requisito é frágil. Com apenas 25 casos de Dermatology, cada fold tem cerca de cinco casos dessa classe; pequenas contagens produzem alterações grandes de recall e F1.

**A conclusão científica mais importante vem dos dados, não de um algoritmo novo.** Existem 730 grupos de transcrições idênticas com rótulos diferentes, envolvendo **1.528 dos 2.669 casos, 57,25%**. Não é apenas duplicação redundante: praticamente todos os grupos repetidos têm categorias concorrentes. Em todos estes 1.528 casos com transcrição não vazia, `description` e `sample_name` também são idênticos entre as cópias.

Consequentemente, a frase atual “although the other fields may distinguish some of those cases” deve ser substituída por uma observação factual: os campos auxiliares não distinguem os conflitos exatos identificados nesta auditoria. Não sabemos se os rótulos são erros de anotação, indexação em várias categorias plausíveis ou uma mistura das duas coisas. O próprio enunciado reconhece rótulos discutíveis e determina que o dataset não será corrigido. Devem preservar a tarefa e discutir a limitação.

Alguns números mostram por que não atribuir tudo ao desequilíbrio:

| Classe | Casos | Casos com transcrição idêntica e outro rótulo | F1 da LR de dois campos, OOF estratificado |
|---|---:|---:|---:|
| Dermatology | 25 | 12, 48,0% | 43,1% |
| Psychiatry-Psychology | 41 | 7, 17,1% | 73,4% |
| Ophthalmology | 73 | 42, 57,5% | 64,1% |
| Neurosurgery | 79 | 72, 91,1% | 27,5% |
| Radiology | 222 | 162, 73,0% | 25,5% |
| Surgery | 893 | 533, 59,7% | 39,9% |

Psychiatry-Psychology tem poucos exemplos e bom F1; Radiology tem muito mais exemplos e é a pior classe desta configuração. A quantidade é uma hipótese explicativa, não uma explicação suficiente. As percentagens de conflito também são descritivas: não isolam causalmente o efeito da ambiguidade, porque classes e conteúdos diferem.

A exclusão de `keywords` tem uma justificação excelente e curta: **todos os 2.219 valores não vazios começam pelo rótulo**, após normalizar capitalização e separadores. Remover apenas o primeiro item não demonstraria que o resto deixa de conter informação derivada do alvo. Para este paper, manter o campo excluído é a decisão mais clara.

Como diagnóstico adicional, agrupar os três campos clínicos por igualdade após `strip()` produz 1.861 inputs distintos. O máximo de acertos de uma única função determinística desses inputs neste corpus seria `sum_g max_c n_gc = 1861`, ou **69,73%**. Inputs idênticos não podem receber previsões distintas de um único classificador que só observa esses campos. Isto é um teto de consistência dos rótulos observados: não é um teto de Bayes, uma previsão para o teste, nem um limite diretamente aplicável ao pooled OOF, que combina modelos diferentes. É uma observação útil na análise de suporte; provavelmente não merece a fórmula e a tabela completa nas três páginas.

A auditoria está reproduzida em [parecer_dados.md](outputs/revisao_cientifica/dados/parecer_dados.md), [auditoria_dados.json](outputs/revisao_cientifica/dados/auditoria_dados.json) e [auditar_dados.py](outputs/revisao_cientifica/dados/auditar_dados.py).

**O protocolo de avaliação muda a pergunta que os scores respondem.** Na CV estratificada por linhas, 1.269 casos, 47,55%, têm o mesmo par título/transcrição no treino do respetivo fold. Em todos eles, as cópias presentes no treino têm exclusivamente outros rótulos. A LR acerta 23,0% nesses casos e 67,6% nos restantes. Estes cortes foram definidos depois de observar os dados completos e servem para diagnóstico, não para selecionar um subconjunto favorável de avaliação.

Assim, é correto dizer que esta CV não mede exclusivamente notas inéditas. Seria incorreto assumir que duplicados tornam sempre o score otimista: aqui, uma cópia com rótulo concorrente pode contrariar o rótulo da linha validada. Também não se deve escolher o protocolo que dá o maior número. A escolha deve corresponder ao cenário: avaliar outras linhas da coleção ou avaliar notas que não apareceram no treino. A documentação de [validação por grupos do scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) explica esta separação.

O novo ensaio manteve a LR de dois campos e alterou apenas o particionamento para `StratifiedGroupKFold`, com a regra de grupos do notebook: transcrição não vazia após `strip()`; ausências com IDs individuais. Obteve **57,21% de accuracy, 59,83% de macro F1 e 45,13% de menor F1, em Neurosurgery**. Todos os cinco folds satisfizeram os dois limiares localmente. É uma condição de avaliação diferente, não uma melhoria de 10,83 pontos causada por um modelo novo.

Foi também executada a baseline com Porter stemming exatamente nos mesmos folds agrupados: **41,29% de accuracy, 15,16% de macro F1 e F1 mínimo zero**. Neste protocolo, a diferença entre as duas configurações é de 15,92 pontos de accuracy. A comparação é justa quanto às partições; continua a comparar configurações completas, com diferenças de algoritmo, campos, representação e pesos.

Há uma limitação residual: esta regra não junta alguns inputs com transcrição vazia e título repetido, deixando seis grupos e doze linhas iguais no par de campos. Não são cinco folds completamente livres de toda a identidade possível de input. Antes da avaliação definitiva, convém definir uma identidade alternativa com os campos disponíveis quando falta transcrição, e reservar IDs individuais para ausência total de input. Devem aplicar a mesma regra a todos os modelos e documentar qualquer mudança; semelhança aproximada entre notas continua fora deste controlo.

**A pergunta atual faz sentido, mas deve ficar mais estreita.** “Which data augmentation strategy most improves…” promete comparar várias estratégias, definir as classes menos representadas e demonstrar melhoria além dos pesos. Até agora, a experiência principal foi a seleção dos campos. A motivação é válida; a resposta ainda não existe.

A formulação recomendada é:

> Does targeted text augmentation improve the trade-off between overall accuracy and per-specialty F1 beyond class weighting in linear clinical-text classification?

Em português: a augmentation textual dirigida melhora o compromisso entre accuracy global e F1 por especialidade, para além dos pesos de classe, na classificação linear de texto clínico?

Esta pergunta aceita uma estratégia textual bem definida, permite uma conclusão negativa e não pressupõe que a pior classe seja a menor. Os conflitos de rótulos passam a ser uma limitação empiricamente demonstrada que ajuda a interpretar a resposta. Se o foco continuar explicitamente nas minorias, definam antecipadamente o subconjunto, por exemplo Dermatology, Psychiatry-Psychology, Ophthalmology e Neurosurgery, todas com menos de 100 exemplos, e reportem também o macro F1 desse subconjunto. Macro F1 das 12 classes não é sinónimo de F1 das quatro minorias.

Uma alternativa, se não conseguirem uma experiência de augmentation suficientemente desenvolvida, é perguntar como a escolha dos campos e o balanceamento afetam o compromisso global/por classe. Fica mais próxima dos resultados já existentes; augmentation continua obrigatória e deve entrar como intervenção adicional avaliada. A primeira opção dá uma história mais focada se concluírem o plano abaixo.

**A literatura apoia escolhas simples, mas não promete um score.** A classificação de notas por subdomínio médico tem precedentes científicos: [Weng et al. (2017)](https://dspace.mit.edu/entities/publication/1c9341f1-179b-4077-baea-37848f979b1f) estudam representações lexicais/conceitos clínicos e classificadores supervisionados, incluindo SVM linear. Isto fundamenta a relevância da tarefa e do terceiro modelo. Os corpora e protocolos são diferentes, pelo que os valores desse artigo não são uma meta transferível para estes dados.

[Wang e Manning (2012)](https://aclanthology.org/P12-2018/) mostram a importância de comparar variantes lineares e representações de forma cuidadosa. É uma razão para justificar unigramas/bigramas e controlar as comparações, não prova de que acrescentar bigramas ou usar SVM seja sempre superior neste problema.

[Wei e Zou (2019)](https://aclanthology.org/D19-1670/) propõem substituição, inserção, troca e remoção de palavras. Os seus ensaios usam CNN/RNN e tarefas gerais, não a vossa LR clínica. O artigo observa danos com perturbações fortes; sustenta uma hipótese de augmentation conservadora, sem garantir benefício ou preservação clínica.

O estudo [Substituting clinical features using synthetic medical phrases (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0933365721001603) motiva o uso de conhecimento e equivalências do domínio, em vez de sinónimos gerais indiscriminados. A revisão de [Feng et al. (2021)](https://aclanthology.org/2021.findings-acl.84/) organiza os mecanismos de augmentation e a sua relação com representação e tarefa. A conclusão prática é escolher uma transformação cujo efeito o modelo consiga observar e cuja preservação do rótulo seja auditável.

**O plano de augmentation deve separar diversidade de repetição.** A minha preferência é começar por uma estratégia lexical conservadora: substituir um número pequeno de abreviaturas ou expressões por equivalentes clínicos contextualizados, usando um dicionário explícito, versionado e auditado. Por exemplo, numa expressão que claramente descreve um exame, uma expansão de “CT” para “computed tomography” muda a forma lexical mantendo o exame referido. Evitem abreviaturas ambíguas, mudanças de diagnóstico, negação, lateralidade, doses ou factos clínicos.

Este exemplo ilustra a regra proposta; não é uma transformação já testada neste projeto. Um dicionário pequeno pode ter cobertura baixa. Por isso, medir cobertura e quantas versões são diferentes na representação é obrigatório. Se quase nenhuma nota elegível mudar, a estratégia tem de ser revista antes de interpretar os scores. Uma adaptação EDA com remoção muito ligeira de palavras genéricas é uma alternativa mais simples, desde que protejam expressões clínicas e negação e auditem exemplos. Não a chamem semanticamente segura sem verificação.

Paráfrases por LLM podem aumentar a diversidade, mas exigem mais validação. Se as usarem, peçam reformulação de uma nota de treino, preservando factos e entidades, em vez de inventar um doente a partir do nome da especialidade. Guardem modelo/versão, prompt, parâmetros, fontes, gerações e rejeições. A especialidade não deve ser introduzida artificialmente no texto como pista da resposta. Não faria geração clínica livre como primeira experiência na véspera da entrega.

Uma comparação mínima interpretável no modelo linear escolhido é:

| Condição | Pesos | Exemplos adicionais | Pergunta respondida |
|---|---|---|---|
| A | Uniformes | Nenhum | Referência sem correção |
| B | Balanced | Nenhum | Efeito dos pesos |
| C | Uniformes | Variantes textuais | Efeito da intervenção textual sem pesos |
| D | Balanced | Variantes textuais | Benefício adicional aos pesos |
| E | Balanced | Cópias dos mesmos exemplos-fonte, no mesmo número de D | Diversidade textual ou apenas mais influência desses exemplos? |

Com os pilotos desta revisão, A e B já têm uma comparação exploratória por linhas; faltam C–E. Na experiência definitiva, **A–E devem usar os mesmos splits, grupos e hiperparâmetros congelados**: os pilotos por linhas não substituem os controlos de uma experiência por grupos. Para a pergunta recomendada, **D versus B** testa o ganho para além dos pesos e **D versus E** testa se a transformação acrescenta algo além da repetição. Controlem o mesmo conjunto de fontes, número de descendentes e critérios de qualidade em D e E. Como ponto de partida executável, proponho uma variante por caso elegível das classes alvo e nenhuma validação aumentada. É um parâmetro inicial a declarar, não um ótimo previsto pela literatura.

Pesos e oversampling não são a mesma intervenção, mas podem ter efeitos muito próximos em modelos lineares. Com features fixas, repetir uma linha equivale a aumentar a sua contribuição na função de perda. Além disso, recalcular automaticamente `balanced` depois de aumentar classes altera os pesos. Para esta experiência, calculem os pesos nos originais de cada fold e usem a mesma regra nos braços comparados. Declarem como controlam a massa adicional e a regularização. Se quiserem isolar diversidade com mais rigor, distribuam o peso de cada exemplo-fonte entre original e descendentes, mantendo o seu peso total; isso é um desenho distinto de oversampling e deve ser apresentado como tal. A [documentação da LR](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) explicita o cálculo de `balanced` e a multiplicação por `sample_weight`.

O fluxo deve ser: dividir originais por grupos; selecionar fontes apenas no treino; gerar/auditar variantes do treino; ajustar representação e classificador apenas nesse treino; prever originais intactos de validação. Original e descendentes nunca podem atravessar folds. Conflitos usados para selecionar ou excluir fontes de augmentation têm de ser identificados apenas com os rótulos do treino de cada fold. A análise de conflitos no corpus completo serve para descrição, não para usar rótulos de validação nessa seleção.

Acrescentar pontuação entre palavras pode não alterar o vetor deste modelo, porque o tokenizador de palavras ignora pontuação; baralhar frases preserva unigramas e quase todos os bigramas internos. A [documentação do TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) permite verificar estas invariâncias. São maus primeiros candidatos se procurarem diversidade lexical útil. SMOTE nos vetores pode ser um comparador metodológico, mas não produz notas clínicas legíveis e não seria a minha primeira escolha para a augmentation textual e respetiva análise exigidas aqui.

Não esperaria que augmentation resolvesse inputs idênticos com rótulos incompatíveis. Pode melhorar cobertura lexical e reduzir dependência de formas superficiais; pode também amplificar ambiguidades, reduzir precisão ou não ter efeito. Qualquer destes resultados, se controlado, responde à pergunta. A conclusão correta de um ganho nulo é restrita à estratégia, orçamento, modelo e protocolo testados.

**A direção dos próximos modelos deve seguir as medições.** LR regularizada sobre TF-IDF continua uma escolha adequada para 2.669 casos, vocabulário especializado e uma tarefa fortemente lexical. Normalizar separadamente o título curto e a nota longa permite que ambos contribuam. Contudo, acrescentar um campo muda dimensionalidade e escala efetiva das features; a ablação atual não prova que `description` seja universalmente inútil, apenas que não ajudou aquela configuração.

Os ensaios diagnósticos desta revisão mantiveram os dois campos e os mesmos folds estratificados. Remover pesos, usar pesos intermédios, aumentar `C` de 1 para 4 e mudar para LinearSVC `C=1` foram piores que o controlo. Assim, a hipótese intuitiva de que remover `balanced` recuperaria accuracy não encontrou apoio. Não recomendo insistir nessa direção como prioridade.

Aumentar `C` reduz a regularização e pode tornar o modelo mais sensível à supervisão contraditória. Por isso, foi incluído um único ensaio adicional com `C=0,25`, decidido depois dos primeiros resultados. Os scores completos são:

| Novos pilotos, CV estratificada por linhas | Accuracy | Macro F1 | Menor F1 |
|---|---:|---:|---:|
| LR balanced, C=1, controlo | 46,38% | 48,96% | 25,45% |
| LR sem pesos, C=1 | 38,22% | 29,72% | 0,00% |
| LR com raiz dos pesos balanced, C=1 | 39,83% | 39,09% | 14,21% |
| LR balanced, C=4 | 40,99% | 43,06% | 14,63% |
| LinearSVC balanced, C=1 | 37,06% | 38,48% | 7,41% |
| **LR balanced, C=0,25** | **48,78%** | **51,81%** | **31,86%** |

O controlo reproduziu exatamente as previsões e os folds da comparação anterior. Reduzir `C` para 0,25 melhorou accuracy e macro F1 em cada um dos cinco folds e F1 em 11 das 12 classes. O modelo acertou 1.302 casos, ficando a seis acertos líquidos dos 1.308 necessários para ultrapassar 49% neste OOF. O ganho não veio simplesmente de recuperar Surgery: o recall dessa classe até desceu de 30,46% para 29,68%. Esta evidência apoia estudar regularização mais forte mantendo os pesos, sem justificar procurar mais valores até obter o número desejado.

**C=0,25 é agora o candidato de desenvolvimento mais promissor no protocolo por linhas.** Não foi executado o cruzamento desse C com a avaliação agrupada; os 57,21% agrupados pertencem a C=1. Não combinar estas duas observações como se fossem uma configuração já medida. LinearSVC é útil como terceiro modelo e comparação de função de perda; a presente configuração não é uma melhoria observada. Um único ensaio também não demonstra que toda a família SVM seja inferior. Os detalhes estão em [LEIA-ME.md](outputs/revisao_cientifica/pilotos/LEIA-ME.md) e nos respetivos CSVs.

Se, depois de fechar protocolo e augmentation, houver necessidade e tempo para outra representação, experimentaria unigramas/bigramas de palavras combinados com n-gramas de caracteres, numa única comparação controlada. Estes podem representar variações de escrita e morfologia; também aumentam dimensão e podem reforçar padrões espúrios. Não é uma garantia de ganho. Só depois consideraria um encoder biomédico. [Gu et al.](https://arxiv.org/abs/2007.15779) fundamentam a utilidade de pré-treino do domínio em tarefas biomédicas, mas não demonstram vantagem neste dataset. Fine-tuning, truncamento de notas, escolha de checkpoint e validação aumentariam muito o trabalho. Para três páginas e este prazo, seria uma extensão opcional, não uma condição para ter um paper bom.

**A seleção e a incerteza precisam de uma regra curta.** Todas as experiências anteriores de campos e estes pilotos são exploração sobre o treino. Reutilizar folds para escolher configurações e publicar o máximo como desempenho independente introduz viés de seleção, como discutem [Cawley e Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html). Trocar só a seed não cria um conjunto de teste independente.

Se houver tempo, usem CV agrupada aninhada, por exemplo cinco folds externos e três internos, com uma grelha pequena definida antes da execução. A seleção de campos e parâmetros que fizer parte do sistema avaliado deve ocorrer no ciclo interno. Conservem o protocolo por linhas como comparação com as experiências antigas. Este desenho controla a seleção automática especificada; não apaga retroativamente todas as decisões humanas motivadas pelo corpus. Se não o conseguirem, descrevam honestamente o caráter exploratório, evitem testes de significância improvisados e limitem as conclusões.

Guardem previsões OOF para comparações emparelhadas e contagens por classe. Para intervalos ou comparações de F1, uma opção é reamostragem emparelhada por grupos de notas; não tratar cópias como observações independentes. Nem esse intervalo elimina o viés da seleção ou toda a dependência entre modelos de CV. [Dror et al. (2018)](https://aclanthology.org/P18-1128/) é uma referência para escolher testes adequados ao desenho e à métrica. Não é necessário ocupar metade da discussão com estatística: a honestidade da interpretação é mais importante do que um p-value acrescentado à pressa.

Para selecionar o sistema de entrega, definam uma regra antes da avaliação final: procurar cumprir os dois limiares e, entre configurações viáveis na validação interna, usar uma métrica principal declarada. Reportem sempre accuracy, macro F1 e F1 mínimo. Se não houver configuração viável, mostrem o compromisso observado; não escondam classes difíceis ou mudem a tarefa.

**Três erros concretos já permitem uma discussão científica curta.** Os exemplos seguintes pertencem ao melhor modelo antigo de dois campos, com OOF estratificado. Não devem ser apresentados como erros do novo modelo agrupado ou de um futuro sistema aumentado sem voltar a verificar as respetivas previsões.

| Caso | Excerto | Rótulo fornecido → previsão | Interpretação sustentada |
|---|---|---|---|
| 302, Excision - Actinic Neoplasm | “Enlarging skin neoplasm, actinic neoplasm, left upper cheek” | Dermatology → Surgery | Pele e excisão são compatíveis com as categorias em competição; o caso 1771 tem os mesmos três campos, rótulo Surgery e previsão Dermatology |
| 911, Facetectomy & Foraminotomy | “Neck pain with bilateral upper extremity radiculopathy” | Neurosurgery → Orthopedic | O caso 1294 é idêntico nos três campos, tem Orthopedic e é previsto como Neurosurgery; o texto sozinho não distingue as duas instâncias |
| 300, MRSA Infection - ER Visit | “A 14-day-old was seen by private doctor because of blister” | Dermatology → General Medicine | Não há cópia exata com rótulo concorrente; conteúdo cutâneo/infeccioso e formato de urgência mostram que os conflitos exatos não explicam tudo |

Escolhi dois exemplos de conflito e um sem conflito para não reduzir toda a discussão a uma única causa. O caso 1548, “CT C-Spine - 2”, Radiology → Neurology, com cópias nas categorias Neurology e Orthopedic, é uma alternativa útil se a pior classe final continuar a ser Radiology. Os excertos, IDs, folds e linhas de origem estão em [erros_oof_exemplos.csv](outputs/revisao_cientifica/dados/erros_oof_exemplos.csv). Uma associação lexical plausível não prova o mecanismo interno; para o afirmar, seria preciso inspecionar contribuições das features ou fazer uma intervenção controlada.

**O limite de três páginas exige substituir texto, não apenas acrescentar.** O relatório atual tem **909 palavras/números de prosa**, por contagem de tokens separados por espaços com pelo menos uma letra ou dígito. Incluindo as células da tabela, são cerca de 968. A tabela ocupa muito mais área tipográfica do que a contagem sugere.

Não existe um número fixo de palavras por página. Para o template de duas colunas, uma página cheia apenas de prosa pode aproximar-se de 500–600 palavras, mas título, autores, secções, tabelas e matriz reduzem muito essa capacidade. Como orçamento de trabalho para este paper, apontaria para **1.000–1.200 palavras de prosa no total**, aproximadamente 330–400 por página em média quando o espaço restante é ocupado pelos elementos exigidos. Só a compilação final confirma a paginação; não foi encontrado compilador LaTeX no PATH nesta revisão.

| Secção | Alvo de palavras de prosa | O que deve fazer |
|---|---:|---|
| Introduction | 90 | Tarefa, problema concreto, pergunta e contribuição |
| Data Analysis | 110 | Dimensão, desequilíbrio, conflitos e exclusão de keywords |
| Data Augmentation | 130 | Transformação, fontes, orçamento, controlo e execução nos folds |
| Models | 145 | NB/LR/SVM, representação e escolhas reproduzíveis |
| Experimental Setup / Parameters | 135 | Grupos, seleção, métricas e parâmetros essenciais |
| Results | 100 | Interpretar a tabela principal e resultados por classe |
| Discussion | 255 | Responder à pergunta, analisar três erros e limitar as conclusões |
| Future Work | 35 | Uma extensão diretamente ligada à limitação principal |
| Declaração de LLMs | 25 | Uso efetivo no código, investigação, escrita e/ou augmentation |
| **Total** | **1.025** | Reservar mais 60–100 palavras para captions; confirmar em LaTeX |

A distribuição por página deve ser pensada em área: primeira página para pergunta, dados, augmentation e início dos modelos; segunda para protocolo, tabela global e resultados por classe; terceira para matriz e discussão. Floats podem mudar esta distribuição. O corpo deve conter uma tabela compacta de sistemas/intervenções, uma tabela por classe do modelo final e uma matriz 12×12 legível. A matriz pode ser de contagens ou normalizada por linha, com denominadores na tabela de classes. Não precisam de duas matrizes lado a lado.

A ablação completa dos sete conjuntos de campos pode ir para o apêndice opcional; no corpo bastam a decisão e dois valores de comparação. Se quiserem apenas três páginas no PDF inteiro, mantenham a ablação detalhada no código/material de suporte e reduzam a prosa para acomodar bibliografia curta. Em qualquer caso, a resposta à pergunta principal e os três erros discutidos ficam no corpo.

Revisões editoriais concretas: reduzir a reparação CSV a uma frase; fundir os três blocos que repetem a superioridade dos dois campos; retirar a diferença de um único acerto entre campos individuais; retirar os caminhos extensos de outputs da narrativa; usar só accuracy/macro F1/F1 mínimo na tabela principal. Precision e recall por classe ajudam a interpretar erros, mas precision/recall macro repetidos em todas as ablações gastam espaço. Não explicar duas páginas de IA na medicina ou definições elementares de métricas.

**Passagens que podem substituir texto atual.** Os números e observações sobre os dados já foram verificados. A abertura apresenta a pergunta e o plano de investigação, cuja experiência de augmentation ainda falta executar; adaptá-la ao estado efetivo do trabalho. Os restantes trechos devem também corresponder à configuração e protocolo efetivamente escolhidos.

Uma abertura curta, sem antecipar resultados de augmentation:

> Clinical specialty classification is complicated by class imbalance and overlapping categories. We investigate whether targeted text augmentation improves the trade-off between overall accuracy and per-specialty F1 beyond class weighting. Our analysis distinguishes limited class support from contradictory supervision: 57.2% of training cases share their transcription with a differently labelled case. We evaluate field-aware linear classifiers; controlled augmentation comparisons will assess whether lexical variation provides additional benefits under these constraints.

Uma substituição para a análise de dados e a possibilidade de os outros campos resolverem os conflitos:

> The repaired training set contains 2,669 cases from 12 specialties, ranging from 25 Dermatology cases to 893 Surgery cases. Exact transcription duplicates carry different labels for 1,528 cases (57.2%); their descriptions and titles are also identical. We retain the supplied labels and distinguish this ambiguity from class imbalance. We exclude keywords because every nonempty value starts with the target specialty.

Uma versão curta da ablação, em lugar dos vários parágrafos atuais:

> With fixed logistic regression settings and stratified folds, sample_name plus transcription achieved the highest observed accuracy (46.4%) and macro F1 (49.0%) among seven field combinations. Adding description reduced these metrics to 45.0% and 47.6%. This exploratory comparison supports the two-field representation under this protocol, without establishing a statistically significant or test-set advantage. Its lowest class F1 was 25.5%.

Uma interpretação fiel ao novo resultado de grupos:

> Keeping repeated nonempty transcriptions within folds increased the same model's accuracy to 57.2%, macro F1 to 59.8% and minimum class F1 to 45.1%. These scores describe a different validation condition, rather than an improved classifier. Cross-label duplicates in row-wise validation help explain why overlap need not inflate performance. Notes without transcriptions require additional identity handling, and official test performance remains unknown.

No texto final, qualquer frase “augmentation improved…” deve indicar comparação, diferença e métrica medidas. Se não houver benefício, uma resposta científica possível é: “Under the tested transformation and augmentation budget, lexical variants did not improve minority-class F1 beyond weighting.” Esta frase é um molde condicional, não um resultado atual.

**A sequência de trabalho recomendada tem cinco decisões.** Primeiro, congelar a pergunta, as métricas e a identidade dos grupos; isso evita que a escolha do protocolo siga o score. Segundo, manter a LR de dois campos balanced e avaliar o candidato C=0,25 no protocolo definitivo, com seleção interna se possível; as variantes sem pesos já não justificam prioridade. Terceiro, executar augmentation e o controlo com cópias nos mesmos folds; isso responde à pergunta que ainda está em aberto. Quarto, construir tabela final, matriz e três erros da configuração final; a discussão deve explicar resultados próprios. Quinto, condensar em LaTeX, verificar as três páginas e alinhar o código de inferência e a entrega.

Há um problema concreto nesse último passo: [classify.py](classify.py) ainda constrói NB sem stemming, [config.py](config.py) mantém apenas `description`, e o notebook principal está configurado com os três campos e `WRITE_ROOT_RESULTS=False`. Por isso, executar o script atual ou entregar o ficheiro existente não demonstra que foi usado o melhor modelo de dois campos. Depois da seleção, a inferência deve reconstruir exatamente o mesmo pré-processamento/modelo, treinar em todo o treino e gerar o `results.txt` pela ordem do leitor validado. Esta revisão não substituiu previsões oficiais.

Na entrega, verificar os contributos percentuais dos autores, nome `NUM.pdf`, ZIP com código e `results.txt` sem cabeçalho, e exclusão dos modelos serializados, conforme o enunciado. Declarar o uso real de LLMs nesta análise, no código e na escrita; se houver geração de dados, documentá-la também. Não atribuir ao sistema utilidade de triagem clínica validada: a evidência é de classificação de categorias deste corpus.

**A previsão mais defensável é que o modelo já tem capacidade para ultrapassar a baseline local e, sob o protocolo por notas agrupadas, os dois limiares.** Não atribuiria uma probabilidade numérica ao teste sem conhecer a sua distribuição e rótulos. A evidência atual favorece concluir e explicar bem um sistema linear, com augmentation controlada, em vez de abrir várias arquiteturas. Um resultado negativo de augmentation, aliado à análise dos conflitos e a uma avaliação correta, pode sustentar um paper completo e cientificamente relevante.

Os ensaios novos são diagnósticos e encontram-se em [pilotos/](outputs/revisao_cientifica/pilotos/), com scripts, métricas por classe/fold, previsões OOF e metadados. O resultado C=0,25 está em [regularizacao_forte/summary.csv](outputs/revisao_cientifica/pilotos/regularizacao_forte/summary.csv), e a baseline agrupada em [grouped_baseline_summary.csv](outputs/revisao_cientifica/pilotos/grouped_baseline_summary.csv). Não substituem automaticamente as tabelas anteriores do relatório.
