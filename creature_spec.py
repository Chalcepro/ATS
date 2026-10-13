"""One creature, read from assets/entities.json and checked.

Everything a creature is lives in the data: its stats, its tags (what it
is), its hostility (why it acts toward the agent), what its hit does to you,
what it leaves when it dies, and what killing it is worth. Game code reads a
CreatureSpec and never asks "is this E020?". A new creature is a new entry.

Unknown keys are an error, so a typo can't silently do nothing. Behaviour
that is written down but not built yet goes under "planned", which nothing
reads; when a planned thing is built it moves out of there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# kind -> the parameters it takes. What each one does is in entities.py.
HOSTILITY = {
    "ignores":           set(),                  # wanders, never attacks
    "flees_when_hit":    set(),                  # wanders, runs from whoever hit it
    "defends_space":     {"space"},              # attacks what comes within `space`
    "strikes_instantly": {"space"},              # adjacent: hits with no warning
    "explodes":          {"space", "radius"},    # telegraphs, then blows up
    "hunts":             set(),                  # moves toward the agent
    "lingering_death":   {"range", "ticks"},     # kills if you stay near too long
}
STATUSES = {"poisoned", "burning", "slowed", "blinded"}
TOP_KEYS = {"name", "brain_id", "island", "tags", "stats", "hostility",
            "on_hit", "on_death", "reward", "killer_name", "planned"}
STAT_KEYS = {"hp", "damage", "telegraph", "move_every"}


@dataclass(frozen=True)
class CreatureSpec:
    id: str
    name: str
    brain_id: int            # the number the brain sees for this creature
    island: int
    tags: frozenset
    hp: int
    damage: int
    telegraph: int
    move_every: int
    hostility: str
    hostility_args: dict
    on_hit: dict             # status -> ticks, applied when its hit lands
    drop: str                # item id, "" for none
    spawn_on_death: tuple    # ((creature id, dx, dy), ...)
    kill_reward: float
    boss: bool
    killer_name: str         # how a death by it is reported
    planned: dict = field(default_factory=dict)

    def has(self, tag: str) -> bool:
        return tag in self.tags


def _only(where: str, got: dict, allowed: set):
    extra = set(got) - allowed
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")


def parse(cid: str, d: dict) -> CreatureSpec:
    w = f"creature {cid}"
    _only(w, d, TOP_KEYS)
    stats = d["stats"]
    _only(w + " stats", stats, STAT_KEYS)
    hos = dict(d["hostility"])
    kind = hos.pop("kind")
    if kind not in HOSTILITY:
        raise ValueError(f"{w}: hostility '{kind}' is not one of {sorted(HOSTILITY)}")
    if set(hos) != HOSTILITY[kind]:
        raise ValueError(f"{w}: hostility '{kind}' takes {sorted(HOSTILITY[kind])}, got {sorted(hos)}")
    _only(w + " on_hit", d.get("on_hit", {}), STATUSES)
    death = d.get("on_death", {})
    _only(w + " on_death", death, {"drop", "spawn"})
    reward = d.get("reward", {})
    _only(w + " reward", reward, {"kill", "boss"})
    return CreatureSpec(
        id=cid, name=d["name"], brain_id=int(d["brain_id"]), island=int(d["island"]),
        tags=frozenset(d.get("tags", ())),
        hp=int(stats["hp"]), damage=int(stats["damage"]),
        telegraph=int(stats["telegraph"]), move_every=int(stats["move_every"]),
        hostility=kind, hostility_args=hos,
        on_hit=dict(d.get("on_hit", {})),
        drop=death.get("drop") or "",
        spawn_on_death=tuple(tuple(s) for s in death.get("spawn", ())),
        kill_reward=float(reward["kill"]), boss=bool(reward.get("boss", False)),
        killer_name=d.get("killer_name", d["name"]),
        planned=dict(d.get("planned", {})),
    )


def parse_all(raw: dict) -> dict:
    specs = {cid: parse(cid, d) for cid, d in raw.items()}
    for s in specs.values():
        for sid, _dx, _dy in s.spawn_on_death:
            if sid not in specs:
                raise ValueError(f"creature {s.id}: spawns unknown creature {sid} on death")
    return specs
