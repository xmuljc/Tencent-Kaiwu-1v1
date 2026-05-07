#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


BUILDING_SUB_TYPES = {21, 22, 24}


class SoldierProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.get_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.feature_config)
        self.one_unit_feature_num = 4
        self.unit_buff_num = 1

    def get_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path)

        self.feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_soldier(self, frame_state, main_hero):
        if main_hero is None:
            return [0.0] * self.one_unit_feature_num

        hero_x = main_hero["location"]["x"]
        hero_z = main_hero["location"]["z"]

        friendly_count = 0
        enemy_count = 0
        nearest_friendly_dist = 30000.0
        nearest_enemy_dist = 30000.0

        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") in BUILDING_SUB_TYPES:
                continue
            if npc.get("hp", 0) <= 0:
                continue

            dist = math.sqrt(
                (npc["location"]["x"] - hero_x) ** 2
                + (npc["location"]["z"] - hero_z) ** 2
            )

            if npc["camp"] == self.main_camp:
                friendly_count += 1
                if dist < nearest_friendly_dist:
                    nearest_friendly_dist = dist
            else:
                enemy_count += 1
                if dist < nearest_enemy_dist:
                    nearest_enemy_dist = dist

        self._friendly_count = friendly_count
        self._enemy_count = enemy_count
        self._nearest_friendly_dist = nearest_friendly_dist
        self._nearest_enemy_dist = nearest_enemy_dist

        vector_feature = []
        for feature_name, feature_func in self.feature_func_map.items():
            value = []
            feature_func(value)
            if feature_name not in self.map_feature_to_norm:
                assert False
            for k in value:
                norm_func, *params = self.map_feature_to_norm[feature_name]
                normalized_value = norm_func(k, *params)
                if isinstance(normalized_value, list):
                    vector_feature.extend(normalized_value)
                else:
                    vector_feature.append(normalized_value)
        return vector_feature

    def get_friendly_soldier_count(self, value):
        value.append(self._friendly_count)

    def get_enemy_soldier_count(self, value):
        value.append(self._enemy_count)

    def get_nearest_friendly_dist(self, value):
        value.append(self._nearest_friendly_dist)

    def get_nearest_enemy_dist(self, value):
        value.append(self._nearest_enemy_dist)
