#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
from agent_ppo.feature.state_adapter import select_nearest_units_by_runtime
import configparser
import os
import math


BUILDING_SUB_TYPES = {21, 22, 24}
SOLDIER_SUB_TYPE = 11
ENEMY_CAMP = {1: 2, 2: 1}


class SoldierProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == 2 or camp == "PLAYERCAMP_2"
        self.get_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.feature_config)
        self.friendly_soldier_slot_num = 4
        self.enemy_soldier_slot_num = 4
        self.soldier_feature_num = 15
        self.enemy_soldier_feature_num = self.soldier_feature_num
        self.one_unit_feature_num = (
            self.friendly_soldier_slot_num + self.enemy_soldier_slot_num
        ) * self.soldier_feature_num

    def get_config(self):
        self.config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path, encoding="utf-8")

        self.feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_soldier(self, frame_state, main_hero, enemy_hero=None):
        if main_hero is None:
            return [0.0] * self.one_unit_feature_num

        friendly_soldiers = []
        enemy_soldiers = []

        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") in BUILDING_SUB_TYPES:
                continue
            if npc.get("sub_type") != SOLDIER_SUB_TYPE:
                continue
            if npc.get("hp", 0) <= 0:
                continue

            if npc["camp"] == self.main_camp:
                friendly_soldiers.append(npc)
            elif npc["camp"] == ENEMY_CAMP.get(self.main_camp):
                enemy_soldiers.append(npc)

        self._friendly_soldiers = select_nearest_units_by_runtime(
            friendly_soldiers, main_hero, self.friendly_soldier_slot_num
        )
        self._enemy_soldiers = select_nearest_units_by_runtime(enemy_soldiers, main_hero, self.enemy_soldier_slot_num)

        vector_feature = []
        vector_feature.extend(
            self._soldier_slot_features(
                self._friendly_soldiers, self.friendly_soldier_slot_num, main_hero, enemy_hero
            )
        )
        vector_feature.extend(
            self._soldier_slot_features(
                self._enemy_soldiers, self.enemy_soldier_slot_num, main_hero, enemy_hero
            )
        )
        return vector_feature

    def _normalize_relative_location(self, diff):
        return max(0.0, min(1.0, (diff + 15000) / 30000.0))

    def _normalize_global_x(self, value):
        return max(0.0, min(1.0, (value + 10000) / 20000.0))

    def _normalize_global_z(self, value):
        return max(0.0, min(1.0, (value + 41000) / 82000.0))

    def _safe_ratio(self, value, max_value):
        if max_value <= 0:
            return 0.0
        return value / max_value

    def _normalize_hero_distance(self, soldier_location, hero):
        if not hero:
            return 1.0
        hero_location = hero.get("location") or {}
        dist = math.sqrt(
            (soldier_location.get("x", 0) - hero_location.get("x", 0)) ** 2
            + (soldier_location.get("z", 0) - hero_location.get("z", 0)) ** 2
        )
        return max(0.0, min(1.0, dist / 30000.0))

    def _is_in_exp_range(self, soldier, main_hero):
        camp_visible = soldier.get("camp_visible")
        if isinstance(camp_visible, (list, tuple)):
            camp_index = 1 if self.transform_camp2_to_camp1 else 0
            if len(camp_visible) > camp_index:
                return 1.0 if camp_visible[camp_index] else 0.0

        sight_area = main_hero.get("sight_area", 0)
        if sight_area <= 0:
            return 0.0
        soldier_location = soldier.get("location") or {}
        hero_location = main_hero.get("location") or {}
        dist = math.sqrt(
            (soldier_location.get("x", 0) - hero_location.get("x", 0)) ** 2
            + (soldier_location.get("z", 0) - hero_location.get("z", 0)) ** 2
        )
        return 1.0 if dist <= sight_area else 0.0

    def _soldier_slot_features(self, soldiers, slot_num, main_hero, enemy_hero):
        vector_feature = []
        hero_x = main_hero["location"]["x"]
        hero_z = main_hero["location"]["z"]
        for soldier in soldiers:
            x_diff = soldier["location"]["x"] - hero_x
            z_diff = soldier["location"]["z"] - hero_z
            if self.transform_camp2_to_camp1:
                x_diff = -x_diff
                z_diff = -z_diff
            location_x = soldier["location"]["x"]
            location_z = soldier["location"]["z"]
            if self.transform_camp2_to_camp1:
                location_x = -location_x
                location_z = -location_z
            vector_feature.extend(
                [
                    self._is_in_exp_range(soldier, main_hero),
                    1.0,
                    1.0 if soldier.get("camp") == self.main_camp else 0.0,
                    self._safe_ratio(soldier.get("hp", 0), soldier.get("max_hp", 0)),
                    self._normalize_relative_location(x_diff),
                    self._normalize_relative_location(z_diff),
                    self._normalize_global_x(location_x),
                    self._normalize_global_z(location_z),
                    self._normalize_hero_distance(soldier["location"], main_hero),
                    self._normalize_hero_distance(soldier["location"], enemy_hero),
                    min(1.0, max(0.0, soldier.get("hp", 0) / 12000.0)),
                    min(1.0, max(0.0, soldier.get("max_hp", 0) / 12000.0)),
                    min(1.0, max(0.0, soldier.get("phy_atk", 0) / 700.0)),
                    min(1.0, max(0.0, soldier.get("kill_income", 0) / 150.0)),
                    1.0 if (soldier.get("buff_state") or {}).get("buff_marks") else 0.0,
                ]
            )

        missing_slot_num = slot_num - len(soldiers)
        vector_feature.extend([0.0] * missing_slot_num * self.soldier_feature_num)
        return vector_feature

    def get_friendly_soldier_count(self, value):
        value.append(self._friendly_count)

    def get_enemy_soldier_count(self, value):
        value.append(self._enemy_count)

    def get_nearest_friendly_dist(self, value):
        value.append(self._nearest_friendly_dist)

    def get_nearest_enemy_dist(self, value):
        value.append(self._nearest_enemy_dist)

    def get_is_in_exp_range(self, value):
        value.append(0.0)

    def get_is_soldier_alive(self, value):
        value.append(1.0)

    def get_belong_to_main_camp(self, value):
        value.append(1.0)

    def get_location_x(self, value):
        value.append(0.0)

    def get_location_z(self, value):
        value.append(0.0)

    def get_dist_to_main_hero(self, value):
        value.append(0.0)

    def get_dist_to_enemy_hero(self, value):
        value.append(0.0)

    def get_hp(self, value):
        value.append(0.0)

    def get_hp_rate(self, value):
        value.append(0.0)

    def get_max_hp(self, value):
        value.append(0.0)

    def get_phy_atk(self, value):
        value.append(0.0)

    def get_kill_income(self, value):
        value.append(0.0)

    def get_buff_marks(self, value):
        value.append(0.0)
