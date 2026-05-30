from collections import OrderedDict
import math
import numpy as np

from agent_ppo.conf.conf import Args
from agent_ppo.feature.state_adapter import (
    CAMP_BLUE,
    CAMP_RED,
    collect_buff_ids,
    normalize_camp,
    safe_get,
    extract_skill_slots,
    adapt_observation,
)


COMMON_BUFF_IDS = [
    90015, 10000, 10010, 10014, 11001, 11002, 11010,
    90025, 911290, 914110, 914210, 914211,
]
LUBAN_BUFF_IDS = [
    112001, 112015, 112025, 112035, 112040, 112043, 112044, 112045, 112046, 112047, 112048,
    112100, 112200, 112201, 112300, 112301, 112320, 112890, 112910, 112990, 112991,
    112000, 112010, 112020, 112030, 112041, 112042, 112190, 112191, 112192, 112210,
]
DIRENJIE_BUFF_IDS = [
    133000, 133001, 133010, 133011, 133020, 133090,
    133200, 133250, 133260, 133950, 133951,
    133100, 133300, 133310,
]
OTHER_OBSERVED_BUFF_IDS = [
    912260, 912262, 912263, 50000, 914230, 914232, 131956, 167600, 167602,
    500009, 801100, 90019, 90110, 911260, 911261, 912330, 919900,
]
BUFF_VOCAB = sorted(set(COMMON_BUFF_IDS + LUBAN_BUFF_IDS + DIRENJIE_BUFF_IDS + OTHER_OBSERVED_BUFF_IDS))
BUFF_TO_INDEX = {buff_id: index for index, buff_id in enumerate(BUFF_VOCAB)}
if len(BUFF_VOCAB) != Args.BUFF_DIM:
    raise ValueError(f"BUFF_DIM mismatch: vocab={len(BUFF_VOCAB)}, config={Args.BUFF_DIM}")

MAX_SOLDIERS = Args.MAX_SOLDIERS
MAX_NEUTRALS = Args.MAX_NEUTRALS
MAX_BULLETS = Args.MAX_BULLETS
MAX_CAKES = Args.MAX_CAKES
MAX_TARGETS = Args.MAX_TARGETS

HERO_DIM = Args.HERO_DIM
TOWER_DIM = Args.TOWER_DIM
SOLDIER_DIM = Args.SOLDIER_DIM
NEUTRAL_DIM = Args.NEUTRAL_DIM
BULLET_DIM = Args.BULLET_DIM
CAKE_DIM = Args.CAKE_DIM
TARGET_DIM = Args.TARGET_DIM
GLOBAL_DIM = Args.GLOBAL_DIM

FEATURE_LAYOUT = OrderedDict(
    [
        ("global", GLOBAL_DIM),
        ("main_hero", HERO_DIM),
        ("enemy_hero", HERO_DIM),
        ("main_tower", TOWER_DIM),
        ("enemy_tower", TOWER_DIM),
        ("soldiers", MAX_SOLDIERS * SOLDIER_DIM),
        ("neutrals", MAX_NEUTRALS * NEUTRAL_DIM),
        ("bullets", MAX_BULLETS * BULLET_DIM),
        ("cakes", MAX_CAKES * CAKE_DIM),
        ("target_candidates", MAX_TARGETS * TARGET_DIM),
        ("buff_multihot", len(BUFF_VOCAB)),
    ]
)

FEATURE_SLICES = {}
_cursor = 0
for _name, _length in FEATURE_LAYOUT.items():
    FEATURE_SLICES[_name] = slice(_cursor, _cursor + _length)
    _cursor += _length
FEATURE_DIM = _cursor


_ADAPTER_NUMERIC_KEYS = {
    "x",
    "y",
    "z",
    "hp",
    "max_hp",
    "cooldown",
    "cooldown_max",
}


def _to_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clip(value, min_value=-10.0, max_value=10.0):
    return max(min_value, min(max_value, _to_float(value)))


def _norm(value, scale, min_value=-10.0, max_value=10.0):
    scale = _to_float(scale)
    if scale == 0:
        return 0.0
    return _clip(_to_float(value) / scale, min_value, max_value)


def _one_hot(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def _safe_location(data):
    location = safe_get(data, "location", {})
    return {
        "x": _to_float(safe_get(location, "x", 100000), 100000.0),
        "y": _to_float(safe_get(location, "y", 0), 0.0),
        "z": _to_float(safe_get(location, "z", 100000), 100000.0),
    }


def _hp_rate(unit):
    hp = _to_float(safe_get(unit, "hp", 0))
    max_hp = _to_float(safe_get(unit, "max_hp", 0))
    return hp / max_hp if max_hp > 0 else 0.0


def _distance_2d(a, b):
    a_loc = _safe_location(a)
    b_loc = _safe_location(b)
    dx = (a_loc["x"] - b_loc["x"]) / 100.0
    dz = (a_loc["z"] - b_loc["z"]) / 100.0
    return math.sqrt(dx * dx + dz * dz)


def _relative_xy(main_hero, unit):
    main_loc = _safe_location(main_hero)
    loc = _safe_location(unit)
    return (loc["x"] - main_loc["x"], loc["z"] - main_loc["z"])


# This follows the existing baseline convention for the 1v1 map: red-side
# observations are rotated by negating x/z so both camps share the same forward
# direction.
def _transform_xz(main_camp, x, z):
    if normalize_camp(main_camp) == CAMP_RED:
        return -_to_float(x), -_to_float(z)
    return _to_float(x), _to_float(z)


def _pad(values, target_len):
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))


def _sanitize_for_adapter(value, key=None):
    if isinstance(value, dict):
        return {item_key: _sanitize_for_adapter(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_adapter(item) for item in value]
    if key in _ADAPTER_NUMERIC_KEYS:
        return _to_float(value)
    return value


class ObservationBuilder:
    def __init__(self, main_camp):
        self.main_camp = normalize_camp(main_camp)

    def reset(self, main_camp):
        self.main_camp = normalize_camp(main_camp)

    def build(self, observation):
        observation = _sanitize_for_adapter(observation)
        state = adapt_observation(observation, main_camp=self.main_camp, player_id=safe_get(observation, "player_id", 0))
        parts = [
            self._global_features(state),
            self._hero_features(state.main_hero, state),
            self._hero_features(state.enemy_hero, state),
            self._tower_features(state.main_tower, state),
            self._tower_features(state.enemy_tower, state),
            self._soldier_features(state),
            self._neutral_features(state),
            self._bullet_features(state),
            self._cake_features(state),
            self._target_features(state),
            self._buff_features(state),
        ]
        feature = []
        for part in parts:
            feature.extend(part)
        feature = [_to_float(x) for x in feature]
        if len(feature) != FEATURE_DIM:
            raise ValueError(f"FEATURE_DIM mismatch: got {len(feature)}, expected {FEATURE_DIM}")
        array = np.asarray(feature, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("feature contains non-finite values")
        return feature

    def _global_features(self, state):
        main = state.main_hero
        enemy = state.enemy_hero
        main_tower = state.main_tower
        enemy_tower = state.enemy_tower
        main_tower_hp = _hp_rate(main_tower)
        enemy_tower_hp = _hp_rate(enemy_tower)
        dist_enemy_tower = _distance_2d(main, enemy_tower) if main and enemy_tower else 0.0
        dist_main_tower = _distance_2d(main, main_tower) if main and main_tower else 0.0
        main_soldiers = [s for s in state.soldiers if normalize_camp(safe_get(s, "camp", 0)) == state.main_camp]
        enemy_soldiers = [s for s in state.soldiers if normalize_camp(safe_get(s, "camp", 0)) == state.enemy_camp]
        values = [
            _norm(state.frame_no, 20000.0, 0.0, 1.5),
            1.0 if state.main_camp == CAMP_BLUE else 0.0,
            1.0 if state.main_camp == CAMP_RED else 0.0,
            *_one_hot(_to_int(safe_get(main, "config_id", 0)), [112, 133]),
            *_one_hot(_to_int(safe_get(enemy, "config_id", 0)), [112, 133]),
            _hp_rate(main),
            _hp_rate(enemy),
            main_tower_hp,
            enemy_tower_hp,
            main_tower_hp - enemy_tower_hp,
            _norm(_to_float(safe_get(main, "money_cnt", safe_get(main, "money", 0))) - _to_float(safe_get(enemy, "money_cnt", safe_get(enemy, "money", 0))), 5000.0),
            _norm(_to_float(safe_get(main, "level", 0)) - _to_float(safe_get(enemy, "level", 0)), 15.0),
            _norm(dist_enemy_tower, 300.0, 0.0, 2.0),
            _norm(dist_main_tower, 300.0, 0.0, 2.0),
            1.0 if enemy else 0.0,
            1.0 if main_tower else 0.0,
            1.0 if enemy_tower else 0.0,
            _norm(len(main_soldiers), 10.0, 0.0, 2.0),
            _norm(len(enemy_soldiers), 10.0, 0.0, 2.0),
            _to_float(state.win),
        ]
        return _pad(values, GLOBAL_DIM)

    def _hero_features(self, hero, state):
        if not hero:
            return [0.0] * HERO_DIM
        loc = _safe_location(hero)
        x, z = _transform_xz(state.main_camp, loc["x"], loc["z"])
        rel_x, rel_z = _relative_xy(state.main_hero, hero) if state.main_hero else (0.0, 0.0)
        rel_x, rel_z = _transform_xz(state.main_camp, rel_x, rel_z)
        values = [
            1.0 if _to_float(safe_get(hero, "hp", 0)) > 0 else 0.0,
            *_one_hot(_to_int(safe_get(hero, "config_id", 0)), [112, 133]),
            _norm(x, 30000.0),
            _norm(z, 30000.0),
            _norm(rel_x, 30000.0),
            _norm(rel_z, 30000.0),
            _hp_rate(hero),
            _norm(safe_get(hero, "level", 0), 15.0, 0.0, 1.5),
            _norm(safe_get(hero, "exp", 0), 2000.0, 0.0, 3.0),
            _norm(safe_get(hero, "money", 0), 5000.0, 0.0, 3.0),
            _norm(safe_get(hero, "money_cnt", 0), 12000.0, 0.0, 3.0),
            _norm(safe_get(hero, "phy_atk", 0), 1000.0, 0.0, 3.0),
            _norm(safe_get(hero, "phy_def", 0), 1000.0, 0.0, 3.0),
            _norm(safe_get(hero, "mgc_def", 0), 1000.0, 0.0, 3.0),
            _norm(safe_get(hero, "mov_spd", 0), 10000.0, 0.0, 2.0),
            _norm(safe_get(hero, "atk_spd", 0), 5000.0, 0.0, 3.0),
            _norm(safe_get(hero, "attack_range", 0), 15000.0, 0.0, 2.0),
            _norm(safe_get(hero, "kill_cnt", 0), 10.0, 0.0, 2.0),
            _norm(safe_get(hero, "dead_cnt", 0), 10.0, 0.0, 2.0),
            _norm(safe_get(hero, "total_hurt_to_hero", 0), 50000.0, 0.0, 5.0),
            _norm(safe_get(hero, "total_be_hurt_by_hero", 0), 50000.0, 0.0, 5.0),
            1.0 if safe_get(hero, "is_in_grass", False) else 0.0,
        ]
        for slot in extract_skill_slots(hero)[:6]:
            values.extend(
                [
                    _norm(safe_get(slot, "slot_type", 0), 10.0, 0.0, 2.0),
                    1.0 if safe_get(slot, "usable", False) else 0.0,
                    _norm(safe_get(slot, "cooldown", 0), max(_to_float(safe_get(slot, "cooldown_max", 1), 1.0), 1.0), 0.0, 1.0),
                    _norm(safe_get(slot, "level", 0), 6.0, 0.0, 2.0),
                ]
            )
        return _pad(values, HERO_DIM)

    def _tower_features(self, tower, state):
        if not tower:
            return [0.0] * TOWER_DIM
        loc = _safe_location(tower)
        x, z = _transform_xz(state.main_camp, loc["x"], loc["z"])
        rel_x, rel_z = _relative_xy(state.main_hero, tower) if state.main_hero else (0.0, 0.0)
        rel_x, rel_z = _transform_xz(state.main_camp, rel_x, rel_z)
        values = [
            1.0 if _to_float(safe_get(tower, "hp", 0)) > 0 else 0.0,
            1.0 if normalize_camp(safe_get(tower, "camp", 0)) == state.main_camp else 0.0,
            1.0 if normalize_camp(safe_get(tower, "camp", 0)) == state.enemy_camp else 0.0,
            _hp_rate(tower),
            _norm(x, 30000.0),
            _norm(z, 30000.0),
            _norm(rel_x, 30000.0),
            _norm(rel_z, 30000.0),
            _norm(_distance_2d(state.main_hero, tower) if state.main_hero else 0.0, 300.0, 0.0, 2.0),
            _norm(safe_get(tower, "attack_range", 0), 15000.0, 0.0, 2.0),
        ]
        return _pad(values, TOWER_DIM)

    def _unit_slot(self, unit, state, dim):
        if not unit:
            return [0.0] * dim
        loc = _safe_location(unit)
        x, z = _transform_xz(state.main_camp, loc["x"], loc["z"])
        rel_x, rel_z = _relative_xy(state.main_hero, unit) if state.main_hero else (0.0, 0.0)
        rel_x, rel_z = _transform_xz(state.main_camp, rel_x, rel_z)
        values = [
            1.0,
            1.0 if normalize_camp(safe_get(unit, "camp", 0)) == state.main_camp else 0.0,
            1.0 if normalize_camp(safe_get(unit, "camp", 0)) == state.enemy_camp else 0.0,
            _hp_rate(unit),
            _norm(x, 30000.0),
            _norm(z, 30000.0),
            _norm(rel_x, 30000.0),
            _norm(rel_z, 30000.0),
            _norm(_distance_2d(state.main_hero, unit) if state.main_hero else 0.0, 300.0, 0.0, 2.0),
            _norm(safe_get(unit, "attack_range", 0), 15000.0, 0.0, 2.0),
        ]
        return _pad(values, dim)

    def _soldier_features(self, state):
        values = []
        for unit in state.soldiers[:MAX_SOLDIERS]:
            values.extend(self._unit_slot(unit, state, SOLDIER_DIM))
        return _pad(values, MAX_SOLDIERS * SOLDIER_DIM)

    def _neutral_features(self, state):
        values = self._unit_slot(state.river_spirit, state, NEUTRAL_DIM)
        return _pad(values, MAX_NEUTRALS * NEUTRAL_DIM)

    def _bullet_features(self, state):
        bullets = []
        for camp in (state.main_camp, state.enemy_camp):
            bullets.extend(state.bullets_by_camp.get(camp, []))
        values = []
        for bullet in bullets[:MAX_BULLETS]:
            loc = _safe_location(bullet)
            x, z = _transform_xz(state.main_camp, loc["x"], loc["z"])
            rel_x, rel_z = _relative_xy(state.main_hero, bullet) if state.main_hero else (0.0, 0.0)
            rel_x, rel_z = _transform_xz(state.main_camp, rel_x, rel_z)
            values.extend(
                _pad(
                    [
                        1.0,
                        1.0 if normalize_camp(safe_get(bullet, "camp", 0)) == state.main_camp else 0.0,
                        1.0 if normalize_camp(safe_get(bullet, "camp", 0)) == state.enemy_camp else 0.0,
                        _norm(safe_get(bullet, "slot_type", 0), 10.0, 0.0, 2.0),
                        _norm(safe_get(bullet, "skill_id", 0), 200000.0, 0.0, 10.0),
                        _norm(x, 30000.0),
                        _norm(z, 30000.0),
                        _norm(rel_x, 30000.0),
                        _norm(rel_z, 30000.0),
                    ],
                    BULLET_DIM,
                )
            )
        return _pad(values, MAX_BULLETS * BULLET_DIM)

    def _cake_features(self, state):
        values = []
        for cake in state.cakes[:MAX_CAKES]:
            collider = safe_get(cake, "collider", {})
            loc = safe_get(collider, "location", {})
            fake_unit = {"location": loc, "hp": 1, "max_hp": 1, "camp": 0, "attack_range": safe_get(collider, "radius", 0)}
            slot = self._unit_slot(fake_unit, state, CAKE_DIM)
            slot[0] = 1.0
            slot[1] = _norm(safe_get(cake, "configId", 0), 10000.0, 0.0, 10.0)
            values.extend(slot)
        return _pad(values, MAX_CAKES * CAKE_DIM)

    def _target_features(self, state):
        candidates = []
        candidates.append(state.enemy_hero)
        candidates.append(state.enemy_tower)
        candidates.extend(state.soldiers)
        candidates.append(state.river_spirit)
        values = []
        for unit in [unit for unit in candidates if unit][:MAX_TARGETS]:
            values.extend(self._unit_slot(unit, state, TARGET_DIM))
        return _pad(values, MAX_TARGETS * TARGET_DIM)

    def _buff_features(self, state):
        values = [0.0] * len(BUFF_VOCAB)
        for unit in (state.main_hero, state.enemy_hero):
            for buff_id in collect_buff_ids(unit):
                index = BUFF_TO_INDEX.get(buff_id)
                if index is not None:
                    values[index] = 1.0
        return values
