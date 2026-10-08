"""Reparação estrutural e auditoria de train.csv, sem alterar os dados de origem.

As regras são específicas da estrutura observada neste dataset. Uma estrutura
desconhecida interrompe a execução em vez de descartar silenciosamente um caso.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

COLUMNS = ["medical_specialty", "description", "sample_name", "transcription", "keywords"]
SPECIALTIES = (
    "Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology", "General Medicine",
    "Neurology", "Neurosurgery", "Obstetrics-Gynecology", "Ophthalmology",
    "Orthopedic", "Psychiatry-Psychology", "Radiology", "Surgery",
)


class DatasetFormatError(ValueError):
    """Formato que as regras de reparação não permitem resolver com segurança."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter=";", strict=True))


def repair_training_csv(source: Path) -> tuple[list[dict], list[dict]]:
    """Repara campos incompletos, colunas vazias excedentes e continuações.

    Mantém a ordem dos casos e os cinco campos originais. A leitura por linha
    física é deliberada: neste ficheiro, cada caso começa por uma especialidade.
    Não é um leitor genérico para CSVs com campos multilinha válidos.
    """
    with source.open(encoding="utf-8-sig", newline="") as stream:
        lines = stream.readlines()
    if not lines or parse_csv_line(lines[0]) != COLUMNS:
        raise DatasetFormatError("O cabeçalho não corresponde aos cinco campos esperados.")

    records: list[dict] = []
    changes: list[dict] = []

    def log(record: dict, line_number: int, action: str, details: str) -> None:
        if action not in record["repair_actions"]:
            record["repair_actions"].append(action)
        changes.append({
            "record_id": record["record_id"], "source_line": line_number,
            "action": action, "details": details,
        })

    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            raise DatasetFormatError(f"Linha {line_number}: linha vazia inesperada.")
        padded_description = False
        try:
            fields = parse_csv_line(line)
        except csv.Error as error:
            body = line.rstrip("\r\n")
            prefix, separator, description = body.partition(";")
            # Sete linhas observadas: especialidade;"descrição, sem mais campos.
            if not (
                prefix in SPECIALTIES and separator and description.startswith('"')
                and body.count('"') == 1 and body.count(";") == 1
            ):
                raise DatasetFormatError(f"Linha {line_number}: {error}") from error
            recovered = parse_csv_line(body + '"')
            if len(recovered) != 2:
                raise DatasetFormatError(f"Linha {line_number}: descrição truncada ambígua.")
            fields = recovered + ["", "", ""]
            padded_description = True

        label = fields[0]
        if label not in SPECIALTIES:
            # Três linhas observadas contêm apenas uma continuação e quatro vazios.
            if not (
                len(fields) == 5 and fields[0].strip() and not any(fields[1:])
                and records and records[-1]["source_line_end"] == line_number - 1
                and "remove_extra_empty_fields" in records[-1]["repair_actions"]
                and records[-1]["transcription"]
            ):
                raise DatasetFormatError(f"Linha {line_number}: especialidade/continuação inesperada.")
            previous = records[-1]
            previous["transcription"] += "\n" + fields[0]
            previous["source_line_end"] = line_number
            previous["is_partial"] = True
            log(previous, line_number, "join_transcription_continuation",
                "Continuação ligada ao caso anterior com uma quebra de linha. "
                "Preservados os caracteres disponíveis; não foram inferidas letras ou palavras.")
            continue

        remove_extra = len(fields) == 7 and fields[5:] == ["", ""]
        if remove_extra:
            fields = fields[:5]
        if len(fields) != len(COLUMNS):
            raise DatasetFormatError(f"Linha {line_number}: {len(fields)} campos em vez de cinco.")
        record = {
            "record_id": len(records) + 1,
            **dict(zip(COLUMNS, fields)),
            "source_line_start": line_number, "source_line_end": line_number,
            "repair_actions": [], "is_partial": padded_description,
        }
        records.append(record)
        if padded_description:
            log(record, line_number, "close_truncated_description",
                "Fechada a aspa da descrição e representados como vazios os três "
                "campos indisponíveis: sample_name, transcription e keywords.")
        if remove_extra:
            log(record, line_number, "remove_extra_empty_fields",
                "Removidos apenas os dois campos vazios excedentes após os cinco campos esperados.")
            if record["transcription"].rstrip().rstrip('"').rstrip().endswith(":"):
                record["is_partial"] = True
                log(record, line_number, "flag_possible_truncation",
                    "A transcrição termina num cabeçalho com dois-pontos; possível texto truncado na origem.")

    # Cada linha com especialidade reconhecida corresponde a um caso recuperado.
    source_labels = [line.partition(";")[0] for line in lines[1:]
                     if line.partition(";")[0] in SPECIALTIES]
    if [record["medical_specialty"] for record in records] != source_labels:
        raise DatasetFormatError("A ordem ou as especialidades dos casos não foram preservadas.")
    represented = [n for record in records
                   for n in range(record["source_line_start"], record["source_line_end"] + 1)]
    if represented != list(range(2, len(lines) + 1)):
        raise DatasetFormatError("Há linhas de origem sem correspondência ou representadas mais de uma vez.")
    return records, changes


def audit_records(records: list[dict]) -> list[dict]:
    """Assinala vazios e repetições, sem eliminar casos ou escolher rótulos."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        text = record["transcription"].strip()
        if text:
            groups[text].append(record)
    audit = []
    for record in records:
        text = record["transcription"].strip()
        group = groups.get(text, [])
        labels = sorted({r["medical_specialty"] for r in group})
        repeated = len(group) > 1
        conflicts = len(labels) > 1
        notes = []
        if record["repair_actions"]:
            notes.append("Registo reparado: " + ", ".join(record["repair_actions"]))
        if record["is_partial"]:
            notes.append("Texto parcial ou potencialmente truncado na origem")
        if not text:
            notes.append("Transcrição vazia")
        if repeated:
            other_ids = ", ".join(str(r["record_id"]) for r in group if r is not record)
            notes.append("Transcrição repetida nos registos " + other_ids)
        if conflicts:
            notes.append("Especialidades diferentes para a mesma transcrição")
        audit.append({
            "record_id": record["record_id"],
            "source_line_start": record["source_line_start"],
            "source_line_end": record["source_line_end"],
            "medical_specialty": record["medical_specialty"],
            "sample_name": record["sample_name"],
            "repair_actions": ";".join(record["repair_actions"]),
            "is_repaired": bool(record["repair_actions"]),
            "is_partial": record["is_partial"],
            **{field + "_empty": not record[field].strip() for field in COLUMNS[1:]},
            "transcription_characters": len(record["transcription"]),
            "duplicate_group_id": hashlib.sha256(text.encode("utf-8")).hexdigest() if repeated else "",
            "duplicate_count": len(group) if repeated else 0,
            "duplicate_record_ids": ";".join(str(r["record_id"]) for r in group) if repeated else "",
            "conflicting_labels": conflicts,
            "group_labels": ";".join(labels) if repeated else "",
            "observacao": "; ".join(notes),
        })
    return audit


def write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_results(root: Path, records: list[dict], changes: list[dict], audit: list[dict]) -> dict:
    """Exporta o CSV, o mapeamento de linhas e relatórios verificáveis."""
    source = root / "train.csv"
    before_hash = sha256(source)
    processed = root / "data" / "processed"
    reports = root / "outputs" / "auditoria"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    clean_path = processed / "train_clean.csv"
    write_csv(clean_path, COLUMNS, records)
    write_csv(reports / "record_audit.csv", list(audit[0]), audit)
    write_csv(reports / "repairs.csv", ["record_id", "source_line", "action", "details"], changes)

    # Releitura estrita: o CSV exportado deve reproduzir todos os campos exatamente.
    with clean_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";", strict=True)
        restored = list(reader)
        if reader.fieldnames != COLUMNS:
            raise DatasetFormatError("O CSV exportado tem um cabeçalho inesperado.")
    expected = [{field: record[field] for field in COLUMNS} for record in records]
    if restored != expected:
        raise DatasetFormatError("A exportação CSV alterou campos ou a ordem dos casos.")
    if sha256(source) != before_hash:
        raise DatasetFormatError("O ficheiro de origem foi alterado.")

    group_ids = {r["duplicate_group_id"] for r in audit if r["duplicate_group_id"]}
    conflicting_ids = {r["duplicate_group_id"] for r in audit if r["conflicting_labels"]}
    summary = {
        "source_file": "train.csv", "source_sha256": before_hash,
        "clean_file": clean_path.relative_to(root).as_posix(), "clean_sha256": sha256(clean_path),
        "records": len(records), "specialties": len({r["medical_specialty"] for r in records}),
        "repair_action_counts": dict(Counter(change["action"] for change in changes)),
        "repaired_records": sum(row["is_repaired"] for row in audit),
        "partial_records": sum(row["is_partial"] for row in audit),
        "empty_fields": {field: sum(row[field + "_empty"] for row in audit) for field in COLUMNS[1:]},
        "duplicate_transcription_groups": len(group_ids),
        "duplicate_transcription_records": sum(bool(row["duplicate_group_id"]) for row in audit),
        "conflicting_label_groups": len(conflicting_ids),
        "conflicting_label_records": sum(row["conflicting_labels"] for row in audit),
        "class_counts": dict(sorted(Counter(r["medical_specialty"] for r in records).items())),
    }
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def export_reader(root: Path, records: list[dict], audit: list[dict], summary: dict) -> Path:
    """Gera um leitor independente para o CSV reparado, com linhas de origem."""
    by_id = {row["record_id"]: row for row in audit}
    payload = {
        "columns": COLUMNS, "labels": list(SPECIALTIES),
        "source_file": summary["clean_file"], "source_sha256": summary["clean_sha256"],
        "original_source_sha256": summary["source_sha256"],
        "records": [{
            "id": r["record_id"], **{field: r[field] for field in COLUMNS},
            "source_line_start": r["source_line_start"], "source_line_end": r["source_line_end"],
            "is_repaired": by_id[r["record_id"]]["is_repaired"],
            "is_partial": r["is_partial"], "observacao": by_id[r["record_id"]]["observacao"],
        } for r in records],
    }
    template = (root / "assets" / "dataset_reader.html").read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("O template do leitor deve conter exatamente um marcador de dados.")
    embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output = root / "outputs" / "dados-legiveis" / "ler_dataset_clean.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__DATA__", embedded), encoding="utf-8")
    return output


def main() -> None:
    """Executa a preparação completa a partir da pasta deste ficheiro."""
    root = Path(__file__).resolve().parent
    records, changes = repair_training_csv(root / "train.csv")
    audit = audit_records(records)
    summary = export_results(root, records, changes, audit)
    reader_path = export_reader(root, records, audit, summary)

    print(f"Dataset criado: {root / summary['clean_file']}")
    print(f"Casos: {summary['records']} | Reparados: {summary['repaired_records']} "
          f"| Parciais: {summary['partial_records']}")
    print(f"Relatórios: {root / 'outputs' / 'auditoria'}")
    print(f"Leitor: {reader_path}")


if __name__ == "__main__":
    main()
