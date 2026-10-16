"""Build a small reproducible ZIP; never include serialized trained models.

Usage: python prepare_submission.py --group 12
Without group/author metadata, --draft creates NUM.zip for local verification
and marks it as unsuitable for submission until the required fields are filled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "augmentation.py", "augmentation_policy.py", "classify.py", "config.py",
    "data_preparation.py", "duplicate_exclusion.py", "models.py",
    "read_test_set.py", "run_final_evaluation.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def readme(group: str, selected: dict, draft: bool) -> str:
    status = "RASCUNHO: preencher os dados do grupo e recompilar antes de submeter.\n\n" if draft else ""
    return status + f"""# Projeto LN — grupo {group}

Sistema selecionado: {selected['candidate_id']}.
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

O PDF de entrega é {group}.pdf. Não há modelos treinados, ambientes Python,
caches ou logs dentro deste ZIP. submission_manifest.json identifica os
ficheiros por hash; os resultados esperados correspondem ao results.txt incluído.
"""


def prepare(group: str | None, draft: bool = False) -> Path:
    import classify
    from augmentation_policy import validate_decision_plan
    from data_preparation import SPECIALTIES

    if group is None and not draft:
        raise ValueError("Indicar --group com o número real do grupo ou usar --draft.")
    name = group if group is not None else "NUM"
    if group is not None and (not group.isdigit() or int(group) < 1):
        raise ValueError("O número do grupo deve ser um inteiro positivo.")
    selected = classify.load_selection()
    plan = validate_decision_plan(ROOT / "augmentation_decision_plan.json", ROOT)
    if selected.get("schema_version") != 3:
        raise ValueError("A seleção ainda não corresponde à experiência de recombinação.")
    source = ROOT / "output/pdf/paper.pdf"
    build = json.loads((ROOT / "output/pdf/compilation.json").read_text(encoding="utf-8-sig"))
    if (build.get("total_pages", 4) > 3 or build.get("bibliography_excluded") is not False
            or build["pdf_sha256"] != digest(source.read_bytes())):
        raise ValueError("Recompilar o PDF atual: no máximo três páginas totais, incluindo bibliografia.")
    for filename, expected in build["source_sha256"].items():
        if digest((ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"O PDF não corresponde à fonte atual: {filename}")
    tex = (ROOT / "paper.tex").read_text(encoding="utf-8")
    placeholder = "[number]" in tex or "[names and contribution percentages]" in tex
    if not draft and (placeholder or f"Group {name}" not in tex):
        raise ValueError("Preencher o grupo/autores/contributos no paper e recompilar.")
    result_data = (ROOT / "results.txt").read_bytes()
    labels = result_data.decode("utf-8").splitlines()
    if len(labels) != 398 or set(labels) - set(SPECIALTIES):
        raise ValueError("results.txt deve conter 398 rótulos válidos, sem cabeçalho.")
    inference = json.loads((ROOT / "outputs/final/inference.json").read_text(encoding="utf-8"))
    if inference["selected_model"] != selected or inference["results_sha256"] != digest(result_data):
        raise ValueError("Os resultados não correspondem à seleção e inferência atuais.")
    if inference["test_metrics"] is not None:
        raise ValueError("Não existem rótulos oficiais para declarar métricas de teste.")
    for filename, key in (("train.csv", "train_sha256"), ("test_no_labels.csv", "test_source_sha256")):
        expected = selected[key] if key == "train_sha256" else inference[key]
        if digest((ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"Dataset incompatível com a inferência: {filename}")
    diagnostics = ROOT / "outputs/final/decoding_diagnostics.csv"
    if digest(diagnostics.read_bytes()) != inference["diagnostics_sha256"]:
        raise ValueError("Os diagnósticos diferem do ficheiro registado na inferência.")
    with diagnostics.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if len(rows) != 398 or [int(row["case_id"]) for row in rows] != list(range(1, 399)):
        raise ValueError("Diagnósticos não estão alinhados com os 398 casos.")
    for label, row in zip(labels, rows):
        if row["predicted_class"] != label or label in json.loads(row["seen_classes"]):
            raise ValueError("Uma previsão não corresponde ao decoder ou repete uma classe excluída.")
    example = Path(plan["review"]["examples_file"])
    if example.is_absolute() or ".." in example.parts:
        raise ValueError("O caminho dos exemplos tem de ser relativo ao projeto.")
    files = {filename: (ROOT / filename).read_bytes() for filename in CODE_FILES}
    for filename in ("selected_model.json", "augmentation_decision_plan.json", "train.csv", "test_no_labels.csv"):
        files[filename] = (ROOT / filename).read_bytes()
    files[example.as_posix()] = (ROOT / example).read_bytes()
    review = ROOT / example.parent / "manual_review.md"
    if review.exists():
        if digest(review.read_bytes()) != plan["review"]["review_sha256"]:
            raise ValueError("A revisão textual difere do registo congelado.")
        files[review.relative_to(ROOT).as_posix()] = review.read_bytes()
    files["requirements.txt"] = (ROOT / "requirements-submission.txt").read_bytes()
    files["README.md"] = readme(name, selected, draft).encode("utf-8")
    files["results.txt"] = result_data
    files[f"{name}.pdf"] = source.read_bytes()
    manifest = {
        "group": name, "ready_for_submission": not draft,
        "candidate_id": selected["candidate_id"], "results_lines": len(labels),
        "results_sha256": digest(result_data), "body_pages": build["body_pages"],
        "total_pages": build["total_pages"], "bibliography_included_in_limit": True,
        "files_sha256": {filename: digest(data) for filename, data in sorted(files.items())},
    }
    files["submission_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output = ROOT / "output/submission"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{name}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for filename, data in sorted(files.items()):
            archive.writestr(filename, data)
    shutil.copyfile(source, output / f"{name}.pdf")
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Falha na verificação CRC do ZIP.")
        for filename, expected in manifest["files_sha256"].items():
            if digest(archive.read(filename)) != expected:
                raise ValueError(f"Conteúdo ZIP divergente: {filename}")
    report = {**manifest, "zip_sha256": digest(target.read_bytes()), "zip": str(target.relative_to(ROOT))}
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("zip", "ready_for_submission", "candidate_id", "results_sha256", "zip_sha256")}, indent=2))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="Número real do grupo.")
    parser.add_argument("--draft", action="store_true", help="Rascunho local, ainda não apto para submissão.")
    args = parser.parse_args()
    prepare(args.group, args.draft)
