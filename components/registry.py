"""Component registry loader for moddable ATS assets."""

import json
from pathlib import Path

import config_rl


class ComponentRegistry:
    def __init__(self, assets_dir=None):
        self.assets_dir = Path(assets_dir or config_rl.ASSETS_DIR)
        self.items = {}
        self.entities = {}
        self.load_all()

    def load_all(self):
        self.items = self._load_json("items.json")
        self.entities = self._load_json("entities.json")
        self._validate()

    def _load_json(self, filename):
        path = self.assets_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _validate(self):
        for item_id, item in self.items.items():
            required = {"name", "type"}
            missing = required - set(item.keys())
            if missing:
                raise ValueError(f"Item {item_id} missing fields: {sorted(missing)}")
        for entity_id, entity in self.entities.items():
            required = {"name", "type", "hp", "damage"}
            missing = required - set(entity.keys())
            if missing:
                raise ValueError(f"Entity {entity_id} missing fields: {sorted(missing)}")

    def get_item(self, item_id):
        return self.items.get(str(item_id))

    def get_entity(self, entity_id):
        return self.entities.get(str(entity_id))

    def item_name(self, item_id):
        item = self.get_item(item_id)
        return item["name"] if item else "Unknown"

    def entity_name(self, entity_id):
        entity = self.get_entity(entity_id)
        return entity["name"] if entity else "Unknown"
