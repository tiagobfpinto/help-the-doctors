# Inspeção qualitativa antes dos ajustes

Revisor: **Codex, inspeção textual; não é validação por um clínico ou por um
investigador humano**. Lidos integralmente os pais, doadores e cinco textos
misturados de `five_variants.json`, antes dos novos ajustes e scores.

Amostragem: os cinco primeiros pais aceites, por classe e ID numérico, no
treino do primeiro fold estratificado, seed 0. Todos são Dermatology por essa
ordenação: esta inspeção não estima a frequência de problemas nas outras
classes nem constitui uma amostra representativa de toda a augmentation.
Não se substituíram exemplos por outros de melhor aparência.

| Pai ← doador | Secção | Observação textual concreta |
|---|---|---|
| 299 ← 322 | TREATMENT | O pai descreve hiperestesia sem alterações visíveis da pele e encaminhamento para neurologia; a variante recebe limpeza de ferida, penso de queimadura e orientações de acidente de trabalho. A história retida não sustenta esse tratamento. O próprio doador alterna 41 e 21 anos, um problema já existente no original. |
| 300 ← 308 | HISTORY OF PRESENT ILLNESS | A história de uma menina de 14 dias com suspeita de MRSA passa a descrever um homem de 34 anos com erupção pruriginosa. Continuam a história de nascimento e exame de uma bebé, incluindo fontanela. Há incompatibilidade explícita de idade, sexo e história. |
| 301 ← 299 | PHYSICAL EXAMINATION | O pai descreve uma mulher com acne/foliculite e afirma ausência de psoríase. O exame inserido usa pronomes masculinos, afirma psoríase/artrite psoriática e hiperestesia. A impressão e tratamento de acne continuam. |
| 302 ← 315 | PROCEDURE | Diagnósticos e lista de procedimentos retidos identificam bochecha, pescoço e orelha; a narrativa inserida opera têmporas e sulco nasolabial. O rótulo amplo é o mesmo, mas anatomia e procedimentos deixam de corresponder entre secções. |
| 304 ← 315 | POSTOPERATIVE DIAGNOSES | Diagnósticos pré-operatórios e técnica descrevem costas/ombro e tratamento nasal; os diagnósticos pós-operatórios passam a indicar têmporas e sulco nasolabial. A mistura contradiz os locais tratados no restante texto. |

O preflight automático, separado desta leitura, verifica nos dez folds que os
pais e doadores são originais do treino, com o mesmo rótulo e grupos de nota
distintos, sem IDs de validação. Recombinação e cópias partilham fontes, número,
rótulos e pesos. Os cinco inputs misturados não coincidem com um original do
treino. Esses controlos não demonstram coerência clínica.

## Decisão antes dos scores

Prosseguir com **uma experiência controlada de mistura textual da mesma
especialidade**, sem alegar preservação semântica ou casos clínicos realistas.
Os factos incompatíveis são uma limitação do método, não foram corrigidos ou
filtrados para melhorar os exemplos. A comparação com cópias permite separar
parte do efeito de repetir fontes do efeito de alterar o texto/TF-IDF.
Um eventual ganho preditivo não provará validade clínica; um resultado negativo
será reportado sem generalizar para todas as técnicas de augmentation.

Mantém-se a regra de promoção escrita em `RECOMBINATION_PLAN.md`: só candidatas
com recombinação são elegíveis; a decisão usa desenvolvimento por linhas com
exclusão e compara com o original da família e o SVC atual. O teste oficial
não foi aberto para definir a transformação ou aprovar esta experiência.

Exemplo curto para o paper, extraído do caso 300 ← 308:
`HISTORY OF PRESENT ILLNESS: A 14-day-old ...` passa a
`HISTORY OF PRESENT ILLNESS: This 34-year-old gentleman ...`, mantendo a história
e exame da bebé. Usar este exemplo para tornar a limitação visível, sem o
apresentar como transformação clinicamente segura.
