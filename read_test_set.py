"""Leitura de test_no_labels.csv alinhada com o results.txt, sem alterar a origem.

O ficheiro é CSV válido: lido com o leitor CSV padrão e reescrito com aspas
mínimas e CRLF, reproduz o original byte a byte. É, portanto, a exportação de uma
tabela com um caso por registo, e é essa tabela que define o alinhamento: a linha
N do results.txt corresponde ao registo CSV N, não à linha física N.

Quatro registos têm uma descrição que ocupa várias linhas físicas. As linhas extra
ainda começam por uma especialidade, ou seja, não eram casos quando a coluna de
rótulos foi removida: são texto de outros casos que ficou dentro da célula. Ficam
fora do texto usado pelo modelo e os seus rótulos nunca são usados.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Sized
from pathlib import Path

from data_preparation import COLUMNS, SPECIALTIES, DatasetFormatError, sha256, write_csv

TEST_COLUMNS = COLUMNS[1:]
# 409 registos CSV em 452 linhas físicas. A confirmar com os docentes.
EXPECTED_TEST_CASES = 409


def read_test_cases(source: Path, expected_cases: int | None = EXPECTED_TEST_CASES) -> list[dict]:
    """Devolve um caso por registo CSV, pela ordem do ficheiro.

    Nenhum registo é removido ou juntado a outro. Estruturas desconhecidas
    interrompem a leitura, tal como na reparação do treino.
    """
    with source.open(encoding="utf-8-sig", newline="") as stream:
        text = stream.read()
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=";", strict=True)
    rows, spans, previous_end = [], [], 0
    try:
        for fields in reader:
            rows.append(fields)
            spans.append((previous_end + 1, reader.line_num))
            previous_end = reader.line_num
    except csv.Error as error:
        raise DatasetFormatError(f"Linha {reader.line_num}: {error}") from error

    # Se a reescrita reproduz o ficheiro, a divisão em registos não é ambígua.
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=";", lineterminator="\r\n").writerows(rows)
    if buffer.getvalue() not in (text, text + "\r\n"):
        raise DatasetFormatError("O ficheiro não é a exportação CSV canónica esperada.")
    if rows and rows[0] == TEST_COLUMNS:
        raise DatasetFormatError("Cabeçalho inesperado: o teste não tem cabeçalho.")

    cases: list[dict] = []
    for fields, (start, end) in zip(rows, spans):
        if len(fields) != len(TEST_COLUMNS):
            raise DatasetFormatError(f"Linha {start}: {len(fields)} campos em vez de quatro.")
        case = {
            "case_id": len(cases) + 1, **dict(zip(TEST_COLUMNS, fields)),
            "source_line_start": start, "source_line_end": end,
            "repair_actions": [], "is_partial": False, "embedded_specialties": "",
        }
        if end > start:
            # Quatro registos observados: descrição própria, depois linhas de outros casos.
            own, *embedded = fields[0].split("\r\n")
            if not (
                len(embedded) == end - start and not any(fields[1:]) and own.strip()
                and ";" not in own
                and all(line.partition(";")[0] in SPECIALTIES for line in embedded)
            ):
                raise DatasetFormatError(f"Linhas {start}-{end}: registo multilinha inesperado.")
            case["description"] = own
            case["is_partial"] = True
            case["embedded_specialties"] = ";".join(line.partition(";")[0] for line in embedded)
            case["repair_actions"].append("separate_embedded_lines")
        elif (
            cases and "separate_embedded_lines" in cases[-1]["repair_actions"]
            and fields[0].strip() and not any(fields[1:])
        ):
            # Quatro registos observados: as keywords da última linha embutida.
            # No treino seriam ligadas ao caso anterior; aqui são um registo da
            # tabela e recebem a sua própria previsão.
            case["is_partial"] = True
            case["repair_actions"].append("flag_continuation_row")
        cases.append(case)

    if expected_cases is not None and len(cases) != expected_cases:
        raise DatasetFormatError(f"{len(cases)} casos em vez de {expected_cases}.")
    return cases


def write_results(path: Path, predictions: list[str], cases: Sized) -> None:
    """Escreve uma especialidade por linha, sem cabeçalho, pela ordem dos casos.

    `cases` é a lista de read_test_cases() ou um DataFrame com as mesmas linhas.
    """
    if len(predictions) != len(cases):
        raise ValueError(f"{len(predictions)} previsões para {len(cases)} casos.")
    unknown = sorted(set(predictions) - set(SPECIALTIES))
    if unknown:
        raise ValueError(f"Especialidades desconhecidas: {unknown}")
    path.write_text("".join(label + "\n" for label in predictions), encoding="utf-8", newline="")


def pandas_counts(source: Path) -> dict | None:
    """Casos obtidos pela leitura ingénua do pandas, para comparação."""
    try:
        import pandas as pd
    except ImportError:
        return None
    return {
        "read_csv_sep_only": len(pd.read_csv(source, sep=";")),
        "read_csv_header_none": len(pd.read_csv(source, sep=";", header=None)),
    }


def main() -> None:
    """Lê o teste, exporta a leitura e a auditoria e compara contagens."""
    root = Path(__file__).resolve().parent
    source = root / "test_no_labels.csv"
    before_hash = sha256(source)
    cases = read_test_cases(source)

    clean_path = root / "data" / "processed" / "test_clean.csv"
    reports = root / "outputs" / "auditoria"
    write_csv(clean_path, TEST_COLUMNS, cases)
    audit = [{
        "case_id": c["case_id"], "source_line_start": c["source_line_start"],
        "source_line_end": c["source_line_end"], "repair_actions": ";".join(c["repair_actions"]),
        "is_partial": c["is_partial"], "embedded_specialties": c["embedded_specialties"],
        **{field + "_empty": not c[field].strip() for field in TEST_COLUMNS},
    } for c in cases]
    write_csv(reports / "test_record_audit.csv", list(audit[0]), audit)
    if sha256(source) != before_hash:
        raise DatasetFormatError("O ficheiro de origem foi alterado.")

    naive = pandas_counts(source)
    summary = {
        "source_file": "test_no_labels.csv", "source_sha256": before_hash,
        "clean_file": clean_path.relative_to(root).as_posix(),
        "physical_lines": cases[-1]["source_line_end"], "cases": len(cases),
        "expected_cases": EXPECTED_TEST_CASES, "pandas_naive_cases": naive,
        "repair_action_counts": dict(Counter(a for c in cases for a in c["repair_actions"])),
        "embedded_lines": sum(c["source_line_end"] - c["source_line_start"] for c in cases),
        "partial_cases": sum(c["is_partial"] for c in cases),
        "empty_fields": {field: sum(row[field + "_empty"] for row in audit) for field in TEST_COLUMNS},
    }
    (reports / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Linhas físicas: {summary['physical_lines']} | Casos (registos CSV): {len(cases)} "
          f"| Esperados: {EXPECTED_TEST_CASES}")
    if naive:
        print(f"pandas read_csv(sep=';'): {naive['read_csv_sep_only']} "
              f"(a primeira linha é tomada como cabeçalho) | header=None: {naive['read_csv_header_none']}")
    for c in cases:
        if c["repair_actions"]:
            print(f"  caso {c['case_id']:3d} (linhas {c['source_line_start']}-{c['source_line_end']}): "
                  f"{', '.join(c['repair_actions'])}")
    print(f"Leitura: {clean_path}")


if __name__ == "__main__":
    main()
