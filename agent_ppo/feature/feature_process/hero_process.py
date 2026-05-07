#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


SUMMONER_SKILL_IDS = {80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121}


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == 2 or camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.get_enemy_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.map_enemy_feature_to_norm = self.normalizer.parse_config(self.enemy_hero_feature_config)
        self.view_dist = 15000
        self.main_unit_feature_num = 18
        self.enemy_unit_feature_num = 11
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def get_enemy_hero_config(self):
        self.enemy_config = configparser.ConfigParser()
        self.enemy_config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "enemy_hero_feature_config.ini")
        self.enemy_config.read(config_path)

        self.enemy_hero_feature_config = []
        for feature, config in self.enemy_config["feature_config"].items():
            self.enemy_hero_feature_config.append(f"{feature}:{config}")

        self.enemy_feature_func_map = {}
        for feature, func_name in self.enemy_config["feature_functions"].items():
            if hasattr(self, func_name):
                self.enemy_feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported enemy function: {func_name}")

    def process_vec_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(
            self.main_camp_hero_dict, "main_camp",
            self.feature_func_map, self.map_feature_to_norm,
            self.main_unit_feature_num
        )
        return main_camp_hero_vector_feature

    def process_vec_enemy_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(
            self.enemy_camp_hero_dict, "enemy_camp",
            self.enemy_feature_func_map, self.map_enemy_feature_to_norm,
            self.enemy_unit_feature_num
        )
        return enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_one_type_hero_feature(self, one_type_hero_info, camp, func_map, norm_map, unit_feature_num):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            for feature_name, feature_func in func_map.items():
                value = []
                feature_func(hero, value, feature_name)
                if feature_name not in norm_map:
                    assert False
                for k in value:
                    norm_func, *params = norm_map[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            for _ in range((self.unit_buff_num - num_heros_considered) * unit_feature_num):
                vector_feature.append(0)
        return vector_feature

    # ========== Main hero features ==========

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_hp_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_level(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("level", 1))

    def get_money(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("money", 0))

    def get_phy_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_atk", 0))

    def get_phy_def(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_def", 0))

    def get_mgc_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_atk", 0))

    def get_mgc_def(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_def", 0))

    def get_mov_spd(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mov_spd", 0))

    def get_kill_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_cnt", 0))

    def get_dead_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("dead_cnt", 0))

    def get_is_in_grass(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("is_in_grass", False):
            value = 1.0
        vector_feature.append(value)

    def _get_skill_cd_rate(self, hero, slot_type):
        skill_state = hero.get("skill_state") or {}
        slot_states = skill_state.get("slot_states") or []
        for slot in slot_states:
            if slot.get("slot_type") == slot_type and slot.get("configId", 0) not in SUMMONER_SKILL_IDS:
                cd_max = slot.get("cooldown_max", 0)
                if cd_max > 0:
                    return slot.get("cooldown", 0) / cd_max
                return 0.0
        return 0.0

    def _get_summoner_cd_rate(self, hero):
        skill_state = hero.get("skill_state") or {}
        slot_states = skill_state.get("slot_states") or []
        for slot in slot_states:
            if slot.get("configId", 0) in SUMMONER_SKILL_IDS:
                cd_max = slot.get("cooldown_max", 0)
                if cd_max > 0:
                    return slot.get("cooldown", 0) / cd_max
                return 0.0
        return 0.0

    def get_skill1_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 1))

    def get_skill2_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 2))

    def get_skill3_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 3))

    def get_summoner_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_summoner_cd_rate(hero))

    # ========== Enemy hero features ==========

    def get_enemy_is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_enemy_hp_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_enemy_level(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("level", 1))

    def get_enemy_phy_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_atk", 0))

    def get_enemy_phy_def(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_def", 0))

    def get_enemy_mgc_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_atk", 0))

    def get_enemy_mgc_def(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_def", 0))

    def get_enemy_mov_spd(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mov_spd", 0))

    def get_enemy_kill_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_cnt", 0))

    def get_enemy_dead_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("dead_cnt", 0))

    def get_enemy_dist(self, hero, vector_feature, feature_name):
        main_hero = self.main_hero_info
        dist = math.sqrt(
            (hero["location"]["x"] - main_hero["location"]["x"]) ** 2
            + (hero["location"]["z"] - main_hero["location"]["z"]) ** 2
        )
        vector_feature.append(dist)
