import glob
import json
import os
import random
import re
import time
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


class Stats(BaseModel):
    APL: str
    Move: str
    Save: str
    Wounds: str


class Weapon(BaseModel):
    name: str
    atk: str
    hit: str
    dmg: str
    rules: str


class RuleEntry(BaseModel):
    id: str = Field(description="Lowercase unique ID. MUST be prefixed with op_, fac_, ploy_, or eq_.")
    category: str = Field(
        description="MUST be exactly one of: 'Operatives', 'Faction Rules', 'Strategy Ploys', 'Firefight Ploys', or 'Faction Equipment'"
    )
    title: str = Field(description="Title case name of the operative, ploy, or rule")
    subtitle: str = Field(description="E.g., 'Leader Operative', 'Operative', 'Faction Rule', 'Strategy Ploy', 'Equipment'")
    stats: Stats | None = Field(default=None)
    weapons: list[Weapon] | None = Field(default=None)
    rules: list[str] = Field(description="Array of strings. Wrap ability/rule names at the start of the string in <b> tags.")
    keywords: str | None = Field(default=None, description="ALL CAPS, comma-separated keywords")


class KillTeamFaction(BaseModel):
    factionName: str = Field(description="The formal name of the Kill Team faction")
    archetypes: list[str] = Field(
        default_factory=list,
        description=(
            "Faction archetypes from the 'ARCHETYPE:' banner on the Operatives/faction card, "
            "normalized to 'Recon', 'Seek & Destroy', 'Security', or 'Infiltration'"
        ),
    )
    rules: list[RuleEntry]


SYSTEM_INSTRUCTION = """
Extract all Warhammer Kill Team rules, operatives, ploys, equipment, and faction archetypes from this PDF.
Look for the 'ARCHETYPE:' banner on the Operatives or faction card (e.g., 'ARCHETYPE: RECON, SEEK & DESTROY') and map each archetype to 'Recon', 'Seek & Destroy', 'Security', or 'Infiltration'.
Format the output perfectly to match the provided JSON schema.
Ignore page numbers, designer commentaries, and flavor text. Focus strictly on playable game rules.
"""

CANONICAL_ARCHETYPES = {
    "recon": "Recon",
    "seek & destroy": "Seek & Destroy",
    "seek and destroy": "Seek & Destroy",
    "security": "Security",
    "infiltration": "Infiltration",
}


def normalize_archetypes(raw_archetypes: list[str] | None) -> list[str]:
    if not raw_archetypes:
        return []
    normalized: list[str] = []
    for arch in raw_archetypes:
        cleaned = re.sub(r"\s+", " ", str(arch or "")).strip().lower()
        canonical = CANONICAL_ARCHETYPES.get(cleaned)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def slugify_faction_stem(pdf_path: str) -> str:
    stem = Path(pdf_path).stem.lower()
    stem = re.sub(r"^eng_\d{2}-\d{2}_kill_team_(?:team_)?rules_", "", stem)
    stem = re.sub(r"_online_rules(?:-[a-z0-9]+)+$", "", stem)
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "unknown_faction"


def classify_error_reason(exc: Exception) -> str:
    msg = str(exc)
    if "503" in msg or "UNAVAILABLE" in msg:
        return "503 UNAVAILABLE (high demand)"
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "429 RATE_LIMIT / QUOTA"
    if isinstance(exc, ValidationError):
        return "Schema validation failed"
    first_line = msg.splitlines()[0] if msg.splitlines() else msg
    return first_line[:70]


def print_summary_table(records: list[dict[str, str]]) -> None:
    if not records:
        print("\nNo PDF files found in pdfs/.")
        return

    headers = ("PDF File", "Status", "Output JSON", "Details")
    rows = [
        (
            r["pdf"],
            r["status"],
            r["output"],
            r["details"],
        )
        for r in records
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cols: tuple[str, ...]) -> str:
        return "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print("\n=== PDF CONVERSION SUMMARY ===")
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)

    succeeded = sum(1 for r in records if r["status"] == "SUCCESS")
    skipped = sum(1 for r in records if r["status"] == "SKIPPED")
    failed = sum(1 for r in records if r["status"] == "FAILED")
    print(f"Totals -> Succeeded: {succeeded} | Skipped: {skipped} | Failed: {failed}\n")


def update_registry(faction_name: str, web_json_path: str, registry_path: str = "factions.json") -> None:
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"factions": []}

    factions_list = registry.setdefault("factions", [])
    existing = next((item for item in factions_list if item.get("path") == web_json_path), None)

    if existing is None:
        factions_list.append({"name": faction_name, "path": web_json_path})
        print(f"Registry added: {faction_name} ({web_json_path})")
    elif existing.get("name") != faction_name:
        existing["name"] = faction_name
        print(f"Registry updated name for {web_json_path}: {faction_name}")

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def generate_with_retry(client, uploaded_file, prompt: str, max_attempts_per_model: int = 2):
    models = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    base_delay = 3.0
    last_exc = None

    for model_idx, model_name in enumerate(models):
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=KillTeamFaction,
                        temperature=0.1,
                    ),
                )
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                is_transient = any(
                    token in msg
                    for token in ("503", "429", "500", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")
                )
                if not is_transient:
                    raise

                if attempt < max_attempts_per_model:
                    sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
                    print(
                        f"[{model_name}] Transient error on attempt {attempt}/{max_attempts_per_model} ({exc}). "
                        f"Retrying in {sleep_for:.1f}s..."
                    )
                    time.sleep(sleep_for)
                else:
                    print(f"[{model_name}] Exhausted {max_attempts_per_model} attempts due to transient error.")

        if model_idx + 1 < len(models):
            print(f"Switching from {model_name} to {models[model_idx + 1]}...")

    raise last_exc


def process_pdfs() -> None:
    client = genai.Client()
    pdf_files = sorted(glob.glob("pdfs/*.pdf"))
    failures: list[tuple[str, str]] = []
    summary_records: list[dict[str, str]] = []

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        stem_slug = slugify_faction_stem(pdf_path)
        json_filename = f"{stem_slug}.json"
        json_path = os.path.join("factions", json_filename)
        web_json_path = f"factions/{json_filename}"

        if os.path.exists(json_path):
            print(f"Skipping existing JSON: {json_path}")
            summary_records.append({
                "pdf": pdf_name,
                "status": "SKIPPED",
                "output": web_json_path,
                "details": "Already exists",
            })
            continue

        print(f"Processing new file: {pdf_path} -> {web_json_path}...")
        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            response = generate_with_retry(
                client,
                uploaded_file,
                "Extract the complete Kill Team faction ruleset, including faction archetypes, from this PDF.",
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            parsed = KillTeamFaction.model_validate_json(response.text)
            normalized_dict = parsed.model_dump()
            normalized_dict["archetypes"] = normalize_archetypes(parsed.archetypes)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(normalized_dict, f, indent=2, ensure_ascii=False)

            update_registry(parsed.factionName, web_json_path)

            arch_label = ", ".join(normalized_dict["archetypes"]) if normalized_dict["archetypes"] else "none"
            summary_records.append({
                "pdf": pdf_name,
                "status": "SUCCESS",
                "output": web_json_path,
                "details": f"{parsed.factionName} ({len(parsed.rules)} entries; {arch_label})",
            })

        except (ValidationError, Exception) as exc:
            reason = classify_error_reason(exc)
            print(f"ERROR processing {pdf_path}: {exc}")
            failures.append((pdf_path, str(exc)))
            summary_records.append({
                "pdf": pdf_name,
                "status": "FAILED",
                "output": web_json_path,
                "details": reason,
            })
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as cleanup_exc:
                    print(f"Warning: failed to delete uploaded file {uploaded_file.name}: {cleanup_exc}")

    print_summary_table(summary_records)

    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} PDF(s): {[f[0] for f in failures]}")


if __name__ == "__main__":
    os.makedirs("factions", exist_ok=True)
    process_pdfs()
