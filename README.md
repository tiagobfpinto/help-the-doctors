# Projeto LN — Grupo 64

`64.pdf` contém o relatório e `results.txt` as 398 previsões para o teste limpo.

Instalar as dependências e gerar novamente as previsões:

```sh
python -m pip install -r requirements.txt
python classify.py
```

O modelo final é um SVC linear com recombinação de secções e exclusão de
rótulos já observados para entradas idênticas no treino. Os CSVs fornecidos
na disciplina estão incluídos para permitir a execução direta.

Para repetir a comparação dos modelos e a validação cruzada do relatório:

```sh
python run_final_evaluation.py
```

Os resultados dessa comparação são guardados em `outputs/avaliacao_recombinacao/`.
Testado com Python 3.14.3 e as versões de `requirements.txt`.
