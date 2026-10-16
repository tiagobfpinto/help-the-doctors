"""Leitura estrita do novo teste, sem cabeçalho e com quatro campos por caso.

Um registo CSV corresponde a uma previsão em results.txt, pela ordem original.
Todos os campos são preservados, incluindo aspas, vazios e quebras de linha
internas válidas. Não há reparação, corte de texto ou interpretação de rótulos.
O ficheiro de origem nunca é alterado.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sized
from pathlib import Path

from data_preparation import COLUMNS, SPECIALTIES, DatasetFormatError, sha256, write_csv

TEST_COLUMNS = COLUMNS[1:]
# Snapshot limpo fornecido pelos docentes, recebido em outubro de 2026.
# A contagem é um contrato do ficheiro atual, não uma regra de reparação.
EXPECTED_TEST_CASES = 398


def read_test_cases(source: Path, expected_cases: int | None = EXPECTED_TEST_CASES) -> list[dict]:
    """Devolve um caso por registo CSV, pela ordem do ficheiro.

    Aceita CSV válido com LF ou CRLF, aspas opcionais e campos multilinha.
    Rejeita cabeçalhos, registos sem quatro campos, CSV inválido e uma contagem
    diferente da esperada. `expected_cases=None` desativa só a verificação da
    contagem, por exemplo para ler fixtures. Nenhum texto é corrigido ou filtrado.
    """
    cases: list[dict] = []
    header_names = set(TEST_COLUMNS)
    previous_end = 0
    try:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter=";", strict=True)
            for fields in reader:
                start, end = previous_end + 1, reader.line_num
                previous_end = end
                if len(fields) != len(TEST_COLUMNS):
                    raise DatasetFormatError(f"Linha {start}: {len(fields)} campos em vez de quatro.")
                if {field.strip().casefold() for field in fields} == header_names:
                    raise DatasetFormatError(f"Linha {start}: cabeçalho inesperado; o teste não tem cabeçalho.")
                cases.append({
                    "case_id": len(cases) + 1, **dict(zip(TEST_COLUMNS, fields)),
                    "source_line_start": start, "source_line_end": end,
                    # Metadados mantidos para os consumidores do leitor anterior.
                    "repair_actions": [], "is_partial": False,
                })
    except csv.Error as error:
        raise DatasetFormatError(f"Linha {reader.line_num}: {error}") from error
    except UnicodeDecodeError as error:
        raise DatasetFormatError("O teste deve estar codificado em UTF-8.") from error
    if not cases:
        raise DatasetFormatError("O ficheiro de teste não contém registos CSV.")
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


def main() -> None:
    """Exporta os registos sem alterações e uma auditoria só de estrutura."""
    root = Path(__file__).resolve().parent
    source = root / "test_no_labels.csv"
    before_hash = sha256(source)
    cases = read_test_cases(source)

    clean_path = root / "data" / "processed" / "test_clean.csv"
    reports = root / "outputs" / "auditoria"
    write_csv(clean_path, TEST_COLUMNS, cases)
    audit = [{
        "case_id": c["case_id"], "source_line_start": c["source_line_start"],
        "source_line_end": c["source_line_end"], "fields": len(TEST_COLUMNS),
    } for c in cases]
    write_csv(reports / "test_record_audit.csv", list(audit[0]), audit)
    if sha256(source) != before_hash:
        raise DatasetFormatError("O ficheiro de origem foi alterado.")

    summary = {
        "source_file": "test_no_labels.csv", "source_sha256": before_hash,
        "source_bytes": source.stat().st_size,
        "clean_file": clean_path.relative_to(root).as_posix(),
        "physical_lines": cases[-1]["source_line_end"], "cases": len(cases),
        "expected_cases": EXPECTED_TEST_CASES, "columns": TEST_COLUMNS,
        "header": False, "delimiter": ";", "repaired_records": 0,
        "multiline_records": sum(c["source_line_end"] > c["source_line_start"] for c in cases),
    }
    (reports / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Linhas físicas: {summary['physical_lines']} | Casos (registos CSV): {len(cases)} "
          f"| Esperados: {EXPECTED_TEST_CASES}")
    print(f"Leitura: {clean_path}")


if __name__ == "__main__":
    main()
