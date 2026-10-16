**Plano registado antes dos novos resultados — 16 de outubro de 2026, 10:05 UTC**

A experiência substitui a remoção de artigos por recombinação de uma secção
entre dois textos da mesma especialidade, exclusivamente entre originais do
treino de cada fold. Os resultados da experiência anterior ficam preservados.
Este plano usa conhecimento exploratório anterior; não é um estudo independente.

A comparação principal usa os mesmos cinco folds estratificados por linhas,
seed 0, com o decoder de exclusão por input completo. A avaliação agrupada usa
os mesmos parâmetros e serve como cenário separado de notas inéditas. NB é a
baseline; LR e SVC são avaliados sem adições, com recombinação e com cópias dos
mesmos exemplos de origem, no mesmo número. Cópias são apenas um controlo.

Uma configuração com recombinação só pode ser promovida se, no protocolo por
linhas com exclusão, cumprir simultaneamente:

- Macro F1 estritamente superior ao original da mesma família.
- F1 de Dermatology estritamente superior ao original da mesma família.
- Queda de accuracy de, no máximo, 0,01 (um ponto percentual).
- Accuracy estritamente superior a 0,49 e F1 mínimo de pelo menos 0,25.

Para substituir o SVC+EXCL atual, a candidata tem ainda de cumprir exatamente
os mesmos critérios face ao SVC+EXCL sem augmentation. Assim, uma melhoria
face à LR não justifica substituir um SVC melhor. Entre candidatas admitidas,
prioriza-se macro F1, depois F1 de Dermatology, accuracy e ID determinístico.
Melhorias estritas usam tolerância numérica de 1e-12. Apenas LR com recombinação
e SVC com recombinação, ambos com exclusão, são elegíveis. Se nenhuma passar,
mantém-se o SVC+EXCL sem augmentation; o resultado negativo integra o paper.

Antes de ajustar modelos, serão inspecionadas cinco variantes determinísticas,
incluindo pai, doador, secção substituída e possíveis incompatibilidades. Não
será alegada coerência de um paciente real ou preservação clínica do significado.
Os marcadores explicativos ficam na auditoria, não nos inputs do modelo, para
não introduzir tokens artificiais sem controlo equivalente. A especificação,
os hashes do código/dados e o registo dessa inspeção serão congelados em
`augmentation_decision_plan.json` antes da execução.

Não se usam o teste oficial, a sua taxa de duplicação ou rótulos privados para
definir a transformação, os parceiros, os parâmetros ou as regras de seleção.
