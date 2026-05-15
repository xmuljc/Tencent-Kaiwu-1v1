import math

from agent_ppo.feature.feature_process.debug_constants import (
    ACTOR_TYPE_MONSTER,
    ACTOR_TYPE_ORGAN,
    BUFFS,
    CAKE_LOCATIONS_BY_CAMP,
    CAMP_BLUE,
    CAMP_RED,
    CONTROL_BUFFS,
    CRYSTAL_CONFIG_IDS,
    DEBUG_FEATURE_DIM,
    DIRENJIE_BUFFS,
    KNOWN_BUFF_SET,
    LANE_SOLDIER_CONFIG_IDS,
    LUBAN_BUFFS,
    RECOVER_BUFFS,
    RIVER_SPIRIT_CONFIG_ID,
    SPRING_TOWER_CONFIG_IDS,
    SUB_TYPE_CRYSTAL,
    SUB_TYPE_LANE_SOLDIER,
    SUB_TYPE_NEUTRAL_MONSTER,
    SUB_TYPE_SPRING_TOWER,
    SUB_TYPE_TOWER,
    TOWER_CONFIG_IDS,
)


class DebugFeatureProcess:
    FEATURE_DIM = DEBUG_FEATURE_DIM

    def __init__(self, camp):
        self.main_camp = camp

    def reset(self, camp):
        self.main_camp = camp

    def process_vec_debug(self, frame_state, main_hero=None, enemy_hero=None):
        if main_hero is None or enemy_hero is None:
            main_hero, enemy_hero = self._find_heroes(frame_state)

        main_buff_ids = self._extract_buff_ids(main_hero)
        enemy_buff_ids = self._extract_buff_ids(enemy_hero)

        features = []
        features.extend(self._multi_hot(main_buff_ids))
        features.extend(self._multi_hot(enemy_buff_ids))
        features.extend(self._hero_identity_features(main_hero, enemy_hero))
        features.extend(self._buff_summary_features(main_buff_ids, enemy_buff_ids))
        features.extend(self._cake_features(frame_state, main_hero, enemy_hero))
        features.extend(self._npc_debug_features(frame_state, main_hero))
        features.extend(self._camp_features())
        features.extend(self._hero_relation_features(main_hero, enemy_hero))

        if len(features) != self.FEATURE_DIM:
            raise RuntimeError(f"debug feature dim mismatch: got {len(features)}, expected {self.FEATURE_DIM}")
        return features

    def _find_heroes(self, frame_state):
        main_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == self.main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _multi_hot(self, buff_ids):
        return [1.0 if buff_id in buff_ids else 0.0 for buff_id in BUFFS]

    def _hero_identity_features(self, main_hero, enemy_hero):
        main_id = self._config_id(main_hero)
        enemy_id = self._config_id(enemy_hero)
        return [
            self._eq(main_id, 112),
            self._eq(main_id, 133),
            self._eq(enemy_id, 112),
            self._eq(enemy_id, 133),
        ]

    def _buff_summary_features(self, main_buff_ids, enemy_buff_ids):
        return [
            self._has_any(main_buff_ids, RECOVER_BUFFS),
            self._has(main_buff_ids, 10000),
            self._has(main_buff_ids, 10010),
            self._has(main_buff_ids, 90015),
            self._has_any(main_buff_ids, CONTROL_BUFFS),
            self._has(main_buff_ids, 11001),
            self._has(main_buff_ids, 11002),
            self._has(main_buff_ids, 11010),
            self._norm_count(main_buff_ids & LUBAN_BUFFS, 12),
            self._norm_count(main_buff_ids & DIRENJIE_BUFFS, 8),
            self._norm_count(main_buff_ids - KNOWN_BUFF_SET, 8),
            self._has_any(enemy_buff_ids, RECOVER_BUFFS),
            self._has(enemy_buff_ids, 10000),
            self._has(enemy_buff_ids, 10010),
            self._has(enemy_buff_ids, 90015),
            self._has_any(enemy_buff_ids, CONTROL_BUFFS),
            self._has(enemy_buff_ids, 11001),
            self._has(enemy_buff_ids, 11002),
            self._has(enemy_buff_ids, 11010),
            self._norm_count(enemy_buff_ids & LUBAN_BUFFS, 12),
            self._norm_count(enemy_buff_ids & DIRENJIE_BUFFS, 8),
            self._norm_count(enemy_buff_ids - KNOWN_BUFF_SET, 8),
        ]

    def _cake_features(self, frame_state, main_hero, enemy_hero):
        main_pos = self._pos(main_hero)
        enemy_pos = self._pos(enemy_hero)
        camp_key = "PLAYERCAMP_2" if self.main_camp == CAMP_RED or self.main_camp == "PLAYERCAMP_2" else "PLAYERCAMP_1"
        own_cake = CAKE_LOCATIONS_BY_CAMP[camp_key]["main"]
        enemy_cake = CAKE_LOCATIONS_BY_CAMP[camp_key]["enemy"]

        nearest_cake_dist = 60000.0
        cake_count = 0
        for cake in frame_state.get("cakes", []):
            cake_count += 1
            cake_pos = self._cake_pos(cake)
            nearest_cake_dist = min(nearest_cake_dist, self._dist_tuple(main_pos, cake_pos))

        main_hp = self._hp_rate(main_hero)
        enemy_hp = self._hp_rate(enemy_hero)
        own_cake_dist = self._dist_tuple(main_pos, own_cake)
        enemy_cake_dist = self._dist_tuple(main_pos, enemy_cake)
        enemy_to_own_cake_dist = self._dist_tuple(enemy_pos, own_cake)
        enemy_to_enemy_cake_dist = self._dist_tuple(enemy_pos, enemy_cake)

        return [
            self._norm_dist(own_cake_dist),
            self._norm_dist(enemy_cake_dist),
            self._norm_dist(enemy_to_own_cake_dist),
            self._norm_dist(enemy_to_enemy_cake_dist),
            self._norm_dist(nearest_cake_dist),
            self._clip(cake_count / 4.0),
            self._bool(main_hp < 0.35 and own_cake_dist < 4000),
            self._bool(main_hp < 0.35 and own_cake_dist >= 4000),
            self._bool(enemy_hp < 0.35 and enemy_to_enemy_cake_dist < 4000),
            self._bool(enemy_hp < 0.35 and enemy_to_enemy_cake_dist >= 4000),
        ]

    def _npc_debug_features(self, frame_state, main_hero):
        main_pos = self._pos(main_hero)
        stats = {
            "friendly_lane_count": 0,
            "enemy_lane_count": 0,
            "friendly_lane_hp": 0.0,
            "enemy_lane_hp": 0.0,
            "nearest_friendly_lane": 60000.0,
            "nearest_enemy_lane": 60000.0,
            "river_alive": 0.0,
            "nearest_river": 60000.0,
            "main_tower_hp": 0.0,
            "enemy_tower_hp": 0.0,
            "main_crystal_hp": 0.0,
            "enemy_crystal_hp": 0.0,
            "nearest_enemy_tower": 60000.0,
            "nearest_enemy_spring": 60000.0,
            "enemy_spring_seen": 0.0,
            "neutral_monster_count": 0,
        }

        for npc in frame_state.get("npc_states", []):
            if npc.get("hp", 0) <= 0:
                continue

            config_id = npc.get("config_id", npc.get("configId", 0))
            sub_type = npc.get("sub_type")
            actor_type = npc.get("actor_type")
            camp = npc.get("camp")
            dist = self._dist_tuple(main_pos, self._pos(npc))
            hp_rate = self._hp_rate(npc)
            is_main_camp = camp == self.main_camp

            if sub_type == SUB_TYPE_LANE_SOLDIER or config_id in LANE_SOLDIER_CONFIG_IDS:
                key_prefix = "friendly" if is_main_camp else "enemy"
                stats[f"{key_prefix}_lane_count"] += 1
                stats[f"{key_prefix}_lane_hp"] += hp_rate
                stats[f"nearest_{key_prefix}_lane"] = min(stats[f"nearest_{key_prefix}_lane"], dist)
                continue

            if config_id == RIVER_SPIRIT_CONFIG_ID or (
                actor_type == ACTOR_TYPE_MONSTER and sub_type == SUB_TYPE_NEUTRAL_MONSTER
            ):
                stats["river_alive"] = 1.0 if config_id == RIVER_SPIRIT_CONFIG_ID else stats["river_alive"]
                stats["nearest_river"] = min(stats["nearest_river"], dist)
                stats["neutral_monster_count"] += 1
                continue

            is_organ = (
                actor_type == ACTOR_TYPE_ORGAN
                or sub_type in {SUB_TYPE_TOWER, SUB_TYPE_SPRING_TOWER, SUB_TYPE_CRYSTAL}
                or config_id in TOWER_CONFIG_IDS
                or config_id in CRYSTAL_CONFIG_IDS
                or config_id in SPRING_TOWER_CONFIG_IDS
            )
            if not is_organ:
                continue

            if sub_type == SUB_TYPE_TOWER or config_id in TOWER_CONFIG_IDS:
                if is_main_camp:
                    stats["main_tower_hp"] = max(stats["main_tower_hp"], hp_rate)
                else:
                    stats["enemy_tower_hp"] = max(stats["enemy_tower_hp"], hp_rate)
                    stats["nearest_enemy_tower"] = min(stats["nearest_enemy_tower"], dist)
                continue

            if sub_type == SUB_TYPE_CRYSTAL or config_id in CRYSTAL_CONFIG_IDS:
                if is_main_camp:
                    stats["main_crystal_hp"] = max(stats["main_crystal_hp"], hp_rate)
                else:
                    stats["enemy_crystal_hp"] = max(stats["enemy_crystal_hp"], hp_rate)
                continue

            if sub_type == SUB_TYPE_SPRING_TOWER or config_id in SPRING_TOWER_CONFIG_IDS:
                if not is_main_camp:
                    stats["enemy_spring_seen"] = 1.0
                    stats["nearest_enemy_spring"] = min(stats["nearest_enemy_spring"], dist)

        return [
            self._clip(stats["friendly_lane_count"] / 10.0),
            self._clip(stats["enemy_lane_count"] / 10.0),
            self._clip(stats["friendly_lane_hp"] / 10.0),
            self._clip(stats["enemy_lane_hp"] / 10.0),
            self._norm_dist(stats["nearest_friendly_lane"]),
            self._norm_dist(stats["nearest_enemy_lane"]),
            stats["river_alive"],
            self._norm_dist(stats["nearest_river"]),
            stats["main_tower_hp"],
            stats["enemy_tower_hp"],
            stats["main_crystal_hp"],
            stats["enemy_crystal_hp"],
            self._norm_dist(stats["nearest_enemy_tower"]),
            stats["enemy_spring_seen"],
            self._norm_dist(stats["nearest_enemy_spring"]),
            self._clip(stats["neutral_monster_count"] / 4.0),
        ]

    def _camp_features(self):
        return [
            self._bool(self.main_camp == CAMP_BLUE or self.main_camp == "PLAYERCAMP_1"),
            self._bool(self.main_camp == CAMP_RED or self.main_camp == "PLAYERCAMP_2"),
        ]

    def _hero_relation_features(self, main_hero, enemy_hero):
        main_pos = self._pos(main_hero)
        enemy_pos = self._pos(enemy_hero)
        rel_x = self._clip((enemy_pos[0] - main_pos[0] + 30000.0) / 60000.0)
        rel_z = self._clip((enemy_pos[1] - main_pos[1] + 30000.0) / 60000.0)
        return [
            self._norm_dist(self._dist_tuple(main_pos, enemy_pos)),
            rel_x,
            rel_z,
            self._clip((self._hp_rate(main_hero) - self._hp_rate(enemy_hero) + 1.0) / 2.0),
            self._clip((self._money(main_hero) - self._money(enemy_hero) + 20000.0) / 40000.0),
            self._clip((self._level(main_hero) - self._level(enemy_hero) + 14.0) / 28.0),
        ]

    def _extract_buff_ids(self, hero):
        if not hero:
            return set()
        buff_state = hero.get("buff_state") or {}
        buff_ids = set()
        for group_name in ("buff_skills", "buff_marks"):
            for buff in buff_state.get(group_name, []) or []:
                buff_id = self._config_id(buff)
                if buff_id:
                    buff_ids.add(buff_id)
        return buff_ids

    @staticmethod
    def _config_id(obj):
        if not obj:
            return 0
        return obj.get("config_id", obj.get("configId", obj.get("id", 0)))

    @staticmethod
    def _hp_rate(obj):
        if not obj:
            return 0.0
        max_hp = obj.get("max_hp", 0)
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, obj.get("hp", 0) / max_hp))

    @staticmethod
    def _money(hero):
        return hero.get("money", 0) if hero else 0

    @staticmethod
    def _level(hero):
        return hero.get("level", 1) if hero else 1

    @staticmethod
    def _pos(obj):
        if not obj:
            return 0.0, 0.0
        location = obj.get("location") or {}
        return float(location.get("x", 0)), float(location.get("z", 0))

    @staticmethod
    def _cake_pos(cake):
        collider = cake.get("collider") or {}
        location = collider.get("location") or cake.get("location") or {}
        return float(location.get("x", 0)), float(location.get("z", 0))

    @staticmethod
    def _dist_tuple(pos_a, pos_b):
        return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    @staticmethod
    def _norm_dist(dist):
        return max(0.0, min(1.0, dist / 60000.0))

    @staticmethod
    def _clip(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _bool(value):
        return 1.0 if value else 0.0

    @staticmethod
    def _eq(value, target):
        return 1.0 if value == target else 0.0

    @staticmethod
    def _has(buff_ids, buff_id):
        return 1.0 if buff_id in buff_ids else 0.0

    @staticmethod
    def _has_any(buff_ids, target_ids):
        return 1.0 if buff_ids & target_ids else 0.0

    @staticmethod
    def _norm_count(values, max_count):
        return max(0.0, min(1.0, len(values) / float(max_count)))
