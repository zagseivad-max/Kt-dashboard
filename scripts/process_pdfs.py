import os
import json
import glob
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
    category: str = Field(description="MUST be exactly one of: 'Operatives', 'Faction Rules', 'Strategy Ploys', 'Firefight Ploys', or 'Faction Equipment'")
    title: str = Field(description="Title case name of the operative, ploy, or rule")
    subtitle: str = Field(description="E.g., 'Leader Operative', 'Operative', 'Faction Rule', 'Strategy Ploy', 'Equipment'")
    stats: Stats | None = Field(default=None)
    weapons: list[Weapon] | None = Field(default=None)
    rules: list[str] = Field(description="Array of strings. Wrap ability/rule names at the start of the string in <b> tags.")
    keywords: str | None = Field(default=None, description="ALL CAPS, comma-separated keywords")

class KillTeamFaction(BaseModel):
    factionName: str = Field(description="The formal name of the Kill Team faction")
    rules: list[RuleEntry]

def update_registry(faction_name, web_json_path):
    registry_path = 'factions.json'
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {"factions": []}
    
    # Avoid duplicates
    if not any(f['path'] == web_json_path for f in registry['factions']):
        registry['factions'].append({"name": faction_name, "path": web_json_path})
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        print(f"Registry updated with: {faction_name}")

def process_pdfs():
    client = genai.Client()
    pdf_files = glob.glob('pdfs/*.pdf')
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        json_filename = filename.replace('.pdf', '.json').replace(' ', '_').lower()
        json_path = os.path.join('factions', json_filename)
        web_json_path = f"factions/{json_filename}"
        
        # Skip if we already converted this PDF
        if os.path.exists(json_path):
            continue
            
        print(f"Processing new file: {filename}...")
        uploaded_file = client.files.upload(file=pdf_path)
        
        prompt = """
        Extract all Warhammer Kill Team rules, operatives, ploys, and equipment from this PDF.
        Format the output perfectly to match the provided JSON schema.
        Ignore page numbers, designer commentaries, and flavor text. Focus strictly on playable game rules.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=KillTeamFaction,
                temperature=0.1,
            ),
        )
        
        raw_json = json.loads(response.text)
        faction_name = raw_json['factionName']
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False)
            
        update_registry(faction_name, web_json_path)
        client.files.delete(name=uploaded_file.name)

if __name__ == "__main__":
    os.makedirs('factions', exist_ok=True)
    process_pdfs()
