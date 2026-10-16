# Projeto LN — grupo 64

Sistema selecionado: SVC_AUG+EXCL.
Testado com Python 3.14.3 e as versões fixadas em requirements.txt.

Na pasta extraída, executar:

    python -m pip install -r requirements.txt
    python classify.py

O segundo comando parte dos 2.669 originais de train.csv e acrescenta as
misturas determinísticas previstas na configuração selecionada. Ajusta o
modelo, constrói o índice só dos originais e gera results.txt: 398 linhas sem cabeçalho,
na ordem de test_no_labels.csv. Não precisa de modelos gravados nem de
ficheiros de validação previamente gerados. Os CSVs fornecidos na disciplina
estão incluídos sem alterações para permitir esta reprodução numa pasta limpa.

Para reproduzir adicionalmente a experiência e a seleção, executar:

    python run_final_evaluation.py
    python classify.py --select

O plano e os cinco exemplos inspecionados estão incluídos. A experiência usa
apenas treino, com cinco folds por linhas e cinco por notas. Compara os modelos
originais, a recombinação de secções e cópias de controlo. A seleção segue a
regra congelada; cópias nunca substituem o sistema final. O teste só gera
previsões e não tem métricas oficiais medidas.

O PDF de entrega é 64.pdf. Não há modelos treinados, ambientes Python,
caches ou logs dentro deste ZIP. submission_manifest.json identifica os
ficheiros por hash; os resultados esperados correspondem ao results.txt incluído.
