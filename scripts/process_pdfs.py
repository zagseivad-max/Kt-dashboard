import glob
import json
import os
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
    rules: list[RuleEntry]


SYSTEM_INSTRUCTION = """
Extract all Warhammer Kill Team rules, operatives, ploys, and equipment from this PDF.
Format the output perfectly to match the provided JSON schema.
Ignore page numbers, designer commentaries, and flavor text. Focus strictly on playable game rules.
"""


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


def process_pdfs() -> None:
    client = genai.Client()
    pdf_files = sorted(glob.glob("pdfs/*.pdf"))
    failures = []

    for pdf_path in pdf_files:
        stem_slug = Path(pdf_path).stem.lower().replace(" ", "_")
        json_filename = f"{stem_slug}.json"
        json_path = os.path.join("factions", json_filename)
        web_json_path = f"factions/{json_filename}"

        if os.path.exists(json_path):
            print(f"Skipping existing JSON: {json_path}")
            continue

        print(f"Processing new file: {pdf_path}...")
        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[uploaded_file, "Extract the complete Kill Team faction ruleset from this PDF."],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=KillTeamFaction,
                    temperature=0.1,
                ),
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            parsed = KillTeamFaction.model_validate_json(response.text)
            normalized_dict = parsed.model_dump()

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(normalized_dict, f, indent=2, ensure_ascii=False)

            update_registry(parsed.factionName, web_json_path)

        except (ValidationError, Exception) as exc:
            print(f"ERROR processing {pdf_path}: {exc}")
            failures.append((pdf_path, str(exc)))
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as cleanup_exc:
                    print(f"Warning: failed to delete uploaded file {uploaded_file.name}: {cleanup_exc}")

    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} PDF(s): {[f[0] for f in failures]}")


if __name__ == "__main__":
    os.makedirs("factions", exist_ok=True)
    process_pdfs()
