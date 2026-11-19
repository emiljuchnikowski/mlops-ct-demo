"""Check a data batch against a YAML data contract."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

NUMERIC = {"int", "float"}


@dataclass
class Violation:
    column: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.column}: {self.detail}"


def load_contract(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate(df: pd.DataFrame, contract: dict) -> list[Violation]:
    """Return every contract violation found in the batch (empty list = batch accepted)."""
    out: list[Violation] = []
    if len(df) < contract.get("min_rows", 0):
        out.append(Violation("*", "min_rows", f"liczba wierszy {len(df)}, wymagane min. {contract['min_rows']}"))

    expected = contract["columns"]
    for col in expected.keys() - set(df.columns):
        out.append(Violation(col, "missing", "brak kolumny w danych"))
    for col in set(df.columns) - expected.keys():
        out.append(Violation(col, "unexpected", "kolumna spoza kontraktu"))

    for col, rule in expected.items():
        if col not in df.columns:
            continue
        s = df[col]
        nulls = int(s.isna().sum())
        if nulls and not rule.get("nullable", True):
            out.append(Violation(col, "nulls", f"puste wartości: {nulls} ({nulls / len(s):.1%}), a kolumna nie może być pusta"))
        s = s.dropna()

        if rule["type"] in NUMERIC and not pd.api.types.is_numeric_dtype(s):
            out.append(Violation(col, "type", f"oczekiwano {rule['type']}, jest {s.dtype}"))
            continue
        if rule["type"] == "int" and not (s % 1 == 0).all():
            out.append(Violation(col, "type", "wartości niecałkowite w kolumnie int"))
        if "min" in rule or "max" in rule:
            lo, hi = rule.get("min", float("-inf")), rule.get("max", float("inf"))
            bad = s[(s < lo) | (s > hi)]
            if len(bad):
                out.append(Violation(col, "range", f"poza zakresem [{lo}, {hi}]: {len(bad)} wartości, np. {bad.iloc[0]}"))
        if "allowed" in rule:
            unknown = set(s.unique()) - set(rule["allowed"])
            if unknown:
                n = int(s.isin(unknown).sum())
                out.append(Violation(col, "allowed", f"nieznane wartości {sorted(map(str, unknown))} (wiersze: {n})"))
        if rule.get("unique") and s.duplicated().any():
            out.append(Violation(col, "unique", f"duplikaty: {int(s.duplicated().sum())}"))
    return out
