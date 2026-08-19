import os
import json
import pandas as pd
from glob import glob

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def inspect_dataset(self):
        """
        Inspects the downloaded dataset files.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")
            
        files = glob(os.path.join(self.data_dir, "*"))
        print(f"Found {len(files)} files in {self.data_dir}")
        for f in files:
            size = os.path.getsize(f)
            print(f"- {os.path.basename(f)} ({size} bytes)")
        
        # Look for JSON/JSONL/CSV files
        json_files = [f for f in files if f.endswith('.json')]
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        csv_files = [f for f in files if f.endswith('.csv')]
        
        return {
            "json": json_files,
            "jsonl": jsonl_files,
            "csv": csv_files
        }

    def load_dataset(self):
        """
        Locates the dataset files, parses them, and returns a normalized internal representation.
        Internal representation: list of examples
        example = {"tokens": [...], "entities": [{"start": int, "end": int, "label": str}], "relations": [{"head": int, "tail": int, "relation": str}]}
        """
        file_types = self.inspect_dataset()
        data = []
        
        # Assumption based on standard Kaggle datasets for NER: json or jsonl are common
        if file_types['json']:
            for f in file_types['json']:
                with open(f, 'r', encoding='utf-8') as file:
                    try:
                        content = json.load(file)
                        if isinstance(content, list):
                            data.extend(content)
                        else:
                            data.append(content)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {f} as standard JSON.")
        elif file_types['jsonl']:
            for f in file_types['jsonl']:
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip():
                            data.append(json.loads(line))
        else:
            print("No JSON or JSONL files found. Attempting to parse CSV if present.")
            # Basic CSV handling fallback
            pass
            
        return self._normalize_data(data)
        
    def _normalize_data(self, raw_data):
        """
        Normalizes dataset to internal dict structure. Needs to adapt to the actual schema.
        Placeholder logic assuming standard nested fields, to be adjusted if dataset differs.
        """
        normalized = []
        for idx, item in enumerate(raw_data):
            # Attempt to find tokens
            tokens = item.get('tokens', item.get('text', []))
            if isinstance(tokens, str):
                tokens = tokens.split() # Simple fallback tokenization if not pre-tokenized
                
            entities = item.get('entities', item.get('ents', []))
            relations = item.get('relations', item.get('rels', []))
            
            norm_entities = []
            for ent in entities:
                # Handle start/end indices. Some datasets use token indices, some use character indices.
                norm_entities.append({
                    "start": ent.get('start', 0),
                    "end": ent.get('end', 0),
                    "label": ent.get('label', ent.get('type', 'UNKNOWN'))
                })
                
            norm_relations = []
            for rel in relations:
                norm_relations.append({
                    "head": rel.get('head', 0),
                    "tail": rel.get('tail', 0),
                    "relation": rel.get('relation', rel.get('type', 'UNKNOWN'))
                })
                
            normalized.append({
                "id": item.get('id', str(idx)),
                "tokens": tokens,
                "entities": norm_entities,
                "relations": norm_relations
            })
            
        return normalized

    def get_entity_distribution(self, normalized_data):
        """
        Produces entity distribution counts.
        """
        counts = {}
        for item in normalized_data:
            for ent in item['entities']:
                label = ent['label']
                counts[label] = counts.get(label, 0) + 1
        return counts
