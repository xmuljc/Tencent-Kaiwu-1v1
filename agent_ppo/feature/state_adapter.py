from dataclasses import dataclass, field
import math


CAMP_BLUE = 1
CAMP_RED = 2
CAMP_NEUTRAL = 0

HERO_CONFIG_IDS = {112, 133}
RIVER_SPIRIT_CONFIG_ID = 6827
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
SPRING_TOWER_CONFIG_IDS = {44, 46}


@dataclass
class AdaptedState:
    frame_no: int
    main_camp: int
    enemy_camp: int
    player_id: int
    win: int = 0
    main_hero: dict = field(default_factory=dict)
    enemy_hero: dict = field(default_factory=dict)
    main_tower: dict = field(default_factory=dict)
    enemy_tower: dict = field(default_factory=dict)
    soldiers: list = field(default_factory=list)
    river_spirit: dict = field(default_factory=dict)
    bullets_by_camp: dict = field(
        default_factory=lambda: {CAMP_BLUE: [], CAMP_RED: [], CAMP_NEUTRAL: []}
    )
    cakes: list = field(default_factory=list)
    death_events: list = field(default_factory=list)


def normalize_camp(camp):
    if camp in (CAMP_BLUE, "1", "CAMP_BLUE", "PLAYERCAMP_1", "blue", "blue_camp"):
        return CAMP_BLUE
    if camp in (CAMP_RED, "2", "CAMP_RED", "PLAYERCAMP_2", "red", "red_camp"):
        return CAMP_RED
    return CAMP_NEUTRAL


def enemy_camp_of(camp):
    camp = normalize_camp(camp)
    if camp == CAMP_BLUE:
        return CAMP_RED
    if camp == CAMP_RED:
        return CAMP_BLUE
    return CAMP_NEUTRAL


def safe_get(data, key, default=0):
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def safe_location(data):
    location = safe_get(data, "location", {})
    return {
        "x": float(safe_get(location, "x", 100000)),
        "y": float(safe_get(location, "y", 0)),
        "z": float(safe_get(location, "z", 100000)),
    }


def hp_rate(unit):
    hp = float(safe_get(unit, "hp", 0))
    max_hp = float(safe_get(unit, "max_hp", 0))
    return hp / max_hp if max_hp > 0 else 0.0


def distance_2d(a, b):
    a_loc = safe_location(a)
    b_loc = safe_location(b)
    dx = (a_loc["x"] - b_loc["x"]) / 100.0
    dz = (a_loc["z"] - b_loc["z"]) / 100.0
    return math.sqrt(dx * dx + dz * dz)


def select_nearest_units_by_runtime(units, reference_unit, limit):
    nearest_units = sorted(
        units,
        key=lambda unit: (
            distance_2d(reference_unit, unit) if reference_unit else 999999.0,
            int(safe_get(unit, "runtime_id", 0)),
        ),
    )[:limit]
    return sorted(nearest_units, key=lambda unit: int(safe_get(unit, "runtime_id", 0)))


def collect_buff_ids(unit):
    buff_state = safe_get(unit, "buff_state", {})
    buff_ids = []
    for item in safe_get(buff_state, "buff_skills", []) or []:
        buff_ids.append(int(safe_get(item, "configId", safe_get(item, "config_id", 0))))
    for item in safe_get(buff_state, "buff_marks", []) or []:
        buff_ids.append(int(safe_get(item, "configId", safe_get(item, "config_id", 0))))
    return [buff_id for buff_id in buff_ids if buff_id > 0]


def extract_skill_slots(hero):
    skill_state = safe_get(hero, "skill_state", {})
    slots = safe_get(skill_state, "slot_states", []) or []
    return sorted(slots, key=lambda slot: int(safe_get(slot, "slot_type", 0)))


def _select_main_hero(hero_states, main_camp, player_id):
    for hero in hero_states:
        if int(safe_get(hero, "runtime_id", -1)) == int(player_id):
            return hero
    for hero in hero_states:
        if normalize_camp(safe_get(hero, "camp", CAMP_NEUTRAL)) == main_camp:
            return hero
    return {}


def _select_enemy_hero(hero_states, enemy_camp):
    for hero in hero_states:
        if normalize_camp(safe_get(hero, "camp", CAMP_NEUTRAL)) == enemy_camp:
            return hero
    return {}


def _is_tower(npc):
    return (
        int(safe_get(npc, "sub_type", -1)) == 21
        or int(safe_get(npc, "config_id", -1)) in TOWER_CONFIG_IDS
    )


def _is_soldier(npc):
    return (
        int(safe_get(npc, "sub_type", -1)) == 11
        or int(safe_get(npc, "config_id", -1)) in LANE_SOLDIER_CONFIG_IDS
    )


def _is_river_spirit(npc):
    return int(safe_get(npc, "config_id", -1)) == RIVER_SPIRIT_CONFIG_ID


def adapt_observation(observation, main_camp=None, player_id=None):
    frame_state = safe_get(observation, "frame_state", observation)
    resolved_player_id = int(
        player_id if player_id is not None else safe_get(observation, "player_id", 0)
    )
    resolved_main_camp = normalize_camp(
        main_camp
        if main_camp is not None
        else safe_get(
            observation,
            "camp",
            safe_get(observation, "player_camp", CAMP_BLUE),
        )
    )
    resolved_enemy_camp = enemy_camp_of(resolved_main_camp)

    hero_states = safe_get(frame_state, "hero_states", []) or []
    npc_states = safe_get(frame_state, "npc_states", []) or []
    bullets = safe_get(frame_state, "bullets", []) or []

    state = AdaptedState(
        frame_no=int(safe_get(frame_state, "frame_no", 0)),
        main_camp=resolved_main_camp,
        enemy_camp=resolved_enemy_camp,
        player_id=resolved_player_id,
        win=int(safe_get(observation, "win", 0)),
        main_hero=_select_main_hero(
            hero_states, resolved_main_camp, resolved_player_id
        ),
        enemy_hero=_select_enemy_hero(hero_states, resolved_enemy_camp),
        cakes=safe_get(frame_state, "cakes", []) or [],
        death_events=safe_get(
            safe_get(frame_state, "frame_action", {}), "dead_action", []
        )
        or [],
    )

    for npc in npc_states:
        npc_camp = normalize_camp(safe_get(npc, "camp", CAMP_NEUTRAL))
        if _is_tower(npc):
            if npc_camp == resolved_main_camp:
                state.main_tower = npc
            elif npc_camp == resolved_enemy_camp:
                state.enemy_tower = npc
        elif _is_soldier(npc):
            state.soldiers.append(npc)
        elif _is_river_spirit(npc):
            state.river_spirit = npc

    main_soldiers = [
        npc
        for npc in state.soldiers
        if normalize_camp(safe_get(npc, "camp", CAMP_NEUTRAL)) == resolved_main_camp
    ]
    enemy_soldiers = [
        npc
        for npc in state.soldiers
        if normalize_camp(safe_get(npc, "camp", CAMP_NEUTRAL)) == resolved_enemy_camp
    ]
    neutral_soldiers = [
        npc
        for npc in state.soldiers
        if normalize_camp(safe_get(npc, "camp", CAMP_NEUTRAL)) == CAMP_NEUTRAL
    ]
    state.soldiers = (
        select_nearest_units_by_runtime(main_soldiers, state.main_hero, 4)
        + select_nearest_units_by_runtime(enemy_soldiers, state.main_hero, 4)
        + select_nearest_units_by_runtime(neutral_soldiers, state.main_hero, 4)
    )


    grouped = {CAMP_BLUE: [], CAMP_RED: [], CAMP_NEUTRAL: []}
    for bullet in bullets:
        grouped.setdefault(
            normalize_camp(safe_get(bullet, "camp", CAMP_NEUTRAL)), []
        ).append(bullet)
    state.bullets_by_camp = grouped
    return state


def adapt_frame_state(frame_state, main_camp, player_id=0, win=0):
    return adapt_observation(
        {
            "frame_state": frame_state,
            "player_id": player_id,
            "camp": main_camp,
            "player_camp": main_camp,
            "win": win,
        },
        main_camp=main_camp,
        player_id=player_id,
    )
