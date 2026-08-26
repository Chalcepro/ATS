from components.registry import ComponentRegistry

REGISTRY = ComponentRegistry()

ITEMS = REGISTRY.items
ENTITIES = REGISTRY.entities


def item_name(item_id):
    return REGISTRY.item_name(item_id)


def entity_name(entity_id):
    return REGISTRY.entity_name(entity_id)
