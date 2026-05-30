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
BULLET_DISTANCE_MAX = 12300.0


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_tower = None
        self.enemy_camp_tower = None
        self.tower_list = []
        self.frd_1v1_cake = None
        self.nearest_enemy_bullet = None
        self.nearest_enemy_tower_bullet = None
        self.nearest_enemy_soldier_bullet = None
        self.enemy_soldiers = []
        self.transform_camp2_to_camp1 = camp == 2 or camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.get_enemy_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.map_enemy_feature_to_norm = self.normalizer.parse_config(self.enemy_hero_feature_config)
        self.view_dist = 15000
        self.main_unit_feature_num = self._feature_num_from_norm_map(
            self.feature_func_map, self.map_feature_to_norm
        )
        self.enemy_unit_feature_num = self._feature_num_from_norm_map(
            self.enemy_feature_func_map, self.map_enemy_feature_to_norm
        )
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path, encoding="utf-8")

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
        self.enemy_config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        self.enemy_config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "enemy_hero_feature_config.ini")
        self.enemy_config.read(config_path, encoding="utf-8")

        self.enemy_hero_feature_config = []
        for feature, config in self.enemy_config["feature_config"].items():
            self.enemy_hero_feature_config.append(f"{feature}:{config}")

        self.enemy_feature_func_map = {}
        for feature, func_name in self.enemy_config["feature_functions"].items():
            if hasattr(self, func_name):
                self.enemy_feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported enemy function: {func_name}")

    def _feature_num_from_norm_map(self, func_map, norm_map):
        feature_num = 0
        for feature_name in func_map:
            if feature_name not in norm_map:
                raise ValueError(f"Missing normalizer config for feature: {feature_name}")
            _, *params = norm_map[feature_name]
            if len(params) == 2 and isinstance(params[0], list):
                feature_num += len(params[0])
            else:
                feature_num += 1
        return feature_num

    @staticmethod
    def _collect_buff_ids(hero):
        buff_state = hero.get("buff_state") or {}
        buff_ids = set()
        for key in ("buff_skills", "buff_marks"):
            for item in buff_state.get(key) or []:
                buff_id = item.get("configId", item.get("config_id", 0))
                try:
                    buff_id = int(buff_id)
                except (TypeError, ValueError):
                    continue
                if buff_id > 0:
                    buff_ids.add(buff_id)
        return buff_ids

    @staticmethod
    def _buff_id_from_feature_name(feature_name):
        return int(feature_name.rsplit("_", 1)[-1])

    def _append_buff_feature(self, hero, vector_feature, feature_name):
        buff_id = self._buff_id_from_feature_name(feature_name)
        vector_feature.append(1.0 if buff_id in self._collect_buff_ids(hero) else 0.0)

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
        self.main_camp_tower = None
        self.enemy_camp_tower = None
        self.tower_list = []
        self.frd_1v1_cake = None
        self.nearest_enemy_bullet = None
        self.nearest_enemy_tower_bullet = None
        self.nearest_enemy_soldier_bullet = None
        self.enemy_soldiers = []
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == 11 and npc.get("camp") != self.main_camp:
                self.enemy_soldiers.append(npc)
                continue
            if npc.get("sub_type") != 21:
                continue
            self.tower_list.append(npc)
            if npc.get("camp") == self.main_camp:
                self.main_camp_tower = npc
            else:
                self.enemy_camp_tower = npc
        self.frd_1v1_cake = self._select_frd_1v1_cake(frame_state.get("cakes", []))
        bullets = frame_state.get("bullets", [])
        enemy_tower_runtime_ids = {
            tower.get("runtime_id") for tower in self.tower_list if tower.get("camp") != self.main_camp
        }
        enemy_soldier_runtime_ids = {soldier.get("runtime_id") for soldier in self.enemy_soldiers}
        self.nearest_enemy_bullet = self._select_nearest_enemy_bullet(bullets)
        self.nearest_enemy_tower_bullet = self._select_nearest_enemy_bullet(
            bullets, source_actor_ids=enemy_tower_runtime_ids
        )
        self.nearest_enemy_soldier_bullet = self._select_nearest_enemy_bullet(
            bullets, source_actor_ids=enemy_soldier_runtime_ids
        )

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

    def get_hp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("hp", 0))

    def get_max_hp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("max_hp", 0))

    def get_hp_recover(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("hp_recover", 0))

    def get_ep(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ep", 0))

    def get_ep_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("max_ep", 0) > 0:
            value = hero.get("ep", 0) / hero.get("max_ep", 0)
        vector_feature.append(value)

    def get_max_ep(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("max_ep", 0))

    def get_ep_recover(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ep_recover", 0))

    def get_level(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("level", 1))

    def get_money(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("money", 0))

    def get_money_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("money_cnt", 0))

    def get_revive_time(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("revive_time", 0))

    def get_kill_income(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_income", 0))

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

    def get_hero_attack_speed(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("atk_spd", 0))

    def get_phy_armor_hurt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_armor_hurt", 0))

    def get_crit_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("crit_rate", 0))

    def get_phy_vamp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_vamp", 0))

    def get_mgc_vamp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_vamp", 0))

    def get_cd_reduce(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("cd_reduce", 0))

    def get_ctrl_reduce(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ctrl_reduce", 0))

    def get_exp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("exp", 0))

    def get_kill_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_cnt", 0))

    def get_dead_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("dead_cnt", 0))

    def get_dist_from_all_heros(self, hero, vector_feature, feature_name):
        enemy_hero = None
        for candidate in self.enemy_camp_hero_dict.values():
            enemy_hero = candidate
            break
        if enemy_hero is None:
            vector_feature.append(98000.0)
            return

        dist = math.sqrt(
            (hero["location"]["x"] - enemy_hero["location"]["x"]) ** 2
            + (hero["location"]["z"] - enemy_hero["location"]["z"]) ** 2
        )
        vector_feature.append(dist)

    def get_is_in_grass(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("is_in_grass", False):
            value = 1.0
        vector_feature.append(value)

    def _distance_xz(self, loc_a, loc_b):
        return math.sqrt((loc_a.get("x", 0) - loc_b.get("x", 0)) ** 2 + (loc_a.get("z", 0) - loc_b.get("z", 0)) ** 2)

    def _is_in_tower_attack_range(self, hero, tower):
        if not tower or tower.get("hp", 0) <= 0:
            return 0.0
        hero_location = hero.get("location") or {}
        tower_location = tower.get("location") or {}
        attack_range = tower.get("attack_range", 0)
        if attack_range <= 0:
            return 0.0
        return 1.0 if self._distance_xz(hero_location, tower_location) <= attack_range else 0.0

    def _is_hero_under_tower_atk(self, hero):
        runtime_id = hero.get("runtime_id", 0)
        if runtime_id == 0:
            return 0.0
        for tower in self.tower_list:
            if tower.get("hp", 0) > 0 and tower.get("attack_target", 0) == runtime_id:
                return 1.0
        return 0.0

    def get_hero_in_main_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_in_tower_attack_range(hero, self.main_camp_tower))

    def get_hero_in_enemy_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_in_tower_attack_range(hero, self.enemy_camp_tower))

    def get_is_hero_under_tower_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_hero_under_tower_atk(hero))

    def _main_camp_cake_point(self):
        if self.transform_camp2_to_camp1:
            return 15340, 15100
        return -15220, -15120

    def _cake_location(self, cake):
        collider = cake.get("collider") or {}
        return collider.get("location") or cake.get("location") or {}

    def _select_frd_1v1_cake(self, cakes):
        if not cakes:
            return None

        cake_x, cake_z = self._main_camp_cake_point()
        selected_cake = None
        selected_dist = float("inf")
        for cake in cakes:
            location = self._cake_location(cake)
            if not location:
                continue
            dist = self._distance_xz(location, {"x": cake_x, "z": cake_z})
            if dist < selected_dist:
                selected_cake = cake
                selected_dist = dist
        if selected_dist > 5000:
            return None
        return selected_cake

    def _frd_1v1_cake_relative_axis(self, hero, axis):
        if not self.frd_1v1_cake:
            return -30000
        cake_location = self._cake_location(self.frd_1v1_cake)
        diff = cake_location.get(axis, 0) - hero.get("location", {}).get(axis, 0)
        if self.transform_camp2_to_camp1:
            diff = -diff
        return diff

    def get_frd_1v1_cake_relative_x(self, hero, vector_feature, feature_name):
        vector_feature.append(self._frd_1v1_cake_relative_axis(hero, "x"))

    def get_frd_1v1_cake_relative_z(self, hero, vector_feature, feature_name):
        vector_feature.append(self._frd_1v1_cake_relative_axis(hero, "z"))

    def get_frd_1v1_cake_exist(self, hero, vector_feature, feature_name):
        vector_feature.append(1.0 if self.frd_1v1_cake else 0.0)

    def _select_nearest_enemy_bullet(self, bullets, source_actor_ids=None):
        main_hero = getattr(self, "main_hero_info", None)
        if not main_hero or not bullets:
            return None

        hero_location = main_hero.get("location") or {}
        selected_bullet = None
        selected_dist = float("inf")
        for bullet in bullets:
            if bullet.get("camp") == self.main_camp:
                continue
            if source_actor_ids is not None and bullet.get("source_actor") not in source_actor_ids:
                continue
            bullet_location = bullet.get("location") or {}
            if not bullet_location:
                continue
            dist = self._distance_xz(hero_location, bullet_location)
            if dist < selected_dist:
                selected_bullet = bullet
                selected_dist = dist
        return selected_bullet

    def _nearest_bullet_relative_axis(self, hero, bullet, axis):
        if not bullet:
            return -30000
        bullet_location = bullet.get("location") or {}
        diff = bullet_location.get(axis, 0) - hero.get("location", {}).get(axis, 0)
        if self.transform_camp2_to_camp1:
            diff = -diff
        return diff

    def _nearest_bullet_distance(self, hero, bullet):
        if not bullet:
            return BULLET_DISTANCE_MAX
        return self._distance_xz(hero.get("location") or {}, bullet.get("location") or {})

    def get_nearest_enemy_bullet_relative_x(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_bullet, "x"))

    def get_nearest_enemy_bullet_relative_z(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_bullet, "z"))

    def get_nearest_enemy_bullet_distance(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_distance(hero, self.nearest_enemy_bullet))

    def get_nearest_enemy_bullet_exist(self, hero, vector_feature, feature_name):
        vector_feature.append(1.0 if self.nearest_enemy_bullet else 0.0)

    def get_nearest_enemy_tower_bullet_relative_x(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_tower_bullet, "x"))

    def get_nearest_enemy_tower_bullet_relative_z(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_tower_bullet, "z"))

    def get_nearest_enemy_tower_bullet_distance(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_distance(hero, self.nearest_enemy_tower_bullet))

    def get_nearest_enemy_tower_bullet_exist(self, hero, vector_feature, feature_name):
        vector_feature.append(1.0 if self.nearest_enemy_tower_bullet else 0.0)

    def get_nearest_enemy_soldier_bullet_relative_x(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_soldier_bullet, "x"))

    def get_nearest_enemy_soldier_bullet_relative_z(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_relative_axis(hero, self.nearest_enemy_soldier_bullet, "z"))

    def get_nearest_enemy_soldier_bullet_distance(self, hero, vector_feature, feature_name):
        vector_feature.append(self._nearest_bullet_distance(hero, self.nearest_enemy_soldier_bullet))

    def get_nearest_enemy_soldier_bullet_exist(self, hero, vector_feature, feature_name):
        vector_feature.append(1.0 if self.nearest_enemy_soldier_bullet else 0.0)

    def _enemy_soldier_in_common_atk_range(self, hero):
        attack_range = hero.get("attack_range", 0)
        if attack_range <= 0:
            return 0.0
        hero_location = hero.get("location") or {}
        for soldier in self.enemy_soldiers:
            if soldier.get("hp", 0) <= 0:
                continue
            soldier_location = soldier.get("location") or {}
            if not soldier_location:
                continue
            if self._distance_xz(hero_location, soldier_location) <= attack_range:
                return 1.0
        return 0.0

    def get_enemy_soldier_in_common_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._enemy_soldier_in_common_atk_range(hero))

    def _enemy_hero_in_common_atk_range(self, hero):
        attack_range = hero.get("attack_range", 0)
        if attack_range <= 0:
            return 0.0
        hero_location = hero.get("location") or {}
        for enemy_hero in self.enemy_camp_hero_dict.values():
            if enemy_hero.get("hp", 0) <= 0:
                continue
            enemy_location = enemy_hero.get("location") or {}
            if not enemy_location:
                continue
            if self._distance_xz(hero_location, enemy_location) <= attack_range:
                return 1.0
        return 0.0

    def get_enemy_hero_in_common_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._enemy_hero_in_common_atk_range(hero))

    def get_enemy_tower_in_common_atk_range(self, hero, vector_feature, feature_name):
        attack_range = hero.get("attack_range", 0)
        tower = self.enemy_camp_tower
        if not tower or tower.get("hp", 0) <= 0 or attack_range <= 0:
            vector_feature.append(0.0)
            return
        vector_feature.append(
            1.0 if self._distance_xz(hero.get("location") or {}, tower.get("location") or {}) <= attack_range else 0.0
        )

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

    def _get_skill_slot_value(self, hero, slot_type, field_name, default=0):
        skill_state = hero.get("skill_state") or {}
        slot_states = skill_state.get("slot_states") or []
        for slot in slot_states:
            if slot.get("slot_type") == slot_type and slot.get("configId", 0) not in SUMMONER_SKILL_IDS:
                value = slot.get(field_name, default)
                if isinstance(value, bool):
                    return 1 if value else 0
                return value
        return default

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

    def _get_skill_cd_by_config_id(self, hero, config_id):
        skill_state = hero.get("skill_state") or {}
        slot_states = skill_state.get("slot_states") or []
        for slot in slot_states:
            if slot.get("configId", 0) == config_id:
                return slot.get("cooldown", 0)
        return 0

    def _get_summoner_cd(self, hero):
        skill_state = hero.get("skill_state") or {}
        slot_states = skill_state.get("slot_states") or []
        for slot in slot_states:
            if slot.get("configId", 0) in SUMMONER_SKILL_IDS:
                return slot.get("cooldown", 0)
        return 0

    def get_skill1_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 1))

    def get_skill2_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 2))

    def get_skill3_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 3))

    def get_summoner_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_summoner_cd_rate(hero))

    def get_skill_1_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 1, "usable"))

    def get_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 1, "cooldown"))

    def get_skill_2_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 2, "usable"))

    def get_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 2, "cooldown"))

    def get_skill_3_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 3, "usable"))

    def get_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 3, "cooldown"))

    def get_heal_skill_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_by_config_id(hero, 90003))

    def get_summon_skill_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_summoner_cd(hero))

    def get_common_skill_is_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 0, "usable"))

    def get_buff(self, hero, vector_feature, feature_name):
        self._append_buff_feature(hero, vector_feature, feature_name)

    # ========== Enemy hero features ==========

    def get_enemy_is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_enemy_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_enemy_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_enemy_hp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("hp", 0))

    def get_enemy_hp_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_enemy_max_hp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("max_hp", 0))

    def get_enemy_hp_recover(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("hp_recover", 0))

    def get_enemy_ep(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ep", 0))

    def get_enemy_ep_rate(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("max_ep", 0) > 0:
            value = hero.get("ep", 0) / hero.get("max_ep", 0)
        vector_feature.append(value)

    def get_enemy_max_ep(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("max_ep", 0))

    def get_enemy_ep_recover(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ep_recover", 0))

    def get_enemy_level(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("level", 1))

    def get_enemy_phy_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_atk", 0))

    def get_enemy_phy_def(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_def", 0))

    def get_enemy_mov_spd(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mov_spd", 0))

    def get_enemy_attack_speed(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("atk_spd", 0))

    def get_enemy_phy_armor_hurt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_armor_hurt", 0))

    def get_enemy_crit_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("crit_rate", 0))

    def get_enemy_phy_vamp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("phy_vamp", 0))

    def get_enemy_mgc_vamp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("mgc_vamp", 0))

    def get_enemy_cd_reduce(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("cd_reduce", 0))

    def get_enemy_ctrl_reduce(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("ctrl_reduce", 0))

    def get_enemy_exp(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("exp", 0))

    def get_enemy_kill_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_cnt", 0))

    def get_enemy_dead_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("dead_cnt", 0))

    def get_enemy_money_cnt(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("money_cnt", 0))

    def get_enemy_money(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("money", 0))

    def get_enemy_revive_time(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("revive_time", 0))

    def get_enemy_kill_income(self, hero, vector_feature, feature_name):
        vector_feature.append(hero.get("kill_income", 0))

    def get_enemy_is_in_grass(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero.get("is_in_grass", False):
            value = 1.0
        vector_feature.append(value)

    def get_enemy_skill1_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 1))

    def get_enemy_skill2_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 2))

    def get_enemy_skill3_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_rate(hero, 3))

    def get_enemy_summoner_cd_rate(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_summoner_cd_rate(hero))

    def get_enemy_skill_1_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 1, "usable"))

    def get_enemy_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 1, "cooldown"))

    def get_enemy_skill_2_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 2, "usable"))

    def get_enemy_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 2, "cooldown"))

    def get_enemy_skill_3_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 3, "usable"))

    def get_enemy_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 3, "cooldown"))

    def get_enemy_heal_skill_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_cd_by_config_id(hero, 90003))

    def get_enemy_summon_skill_cd(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_summoner_cd(hero))

    def get_enemy_common_skill_is_useable(self, hero, vector_feature, feature_name):
        vector_feature.append(self._get_skill_slot_value(hero, 0, "usable"))

    def get_enemy_buff(self, hero, vector_feature, feature_name):
        self._append_buff_feature(hero, vector_feature, feature_name)

    def get_enemy_hero_in_main_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_in_tower_attack_range(hero, self.main_camp_tower))

    def get_enemy_hero_in_enemy_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_in_tower_attack_range(hero, self.enemy_camp_tower))

    def get_enemy_is_hero_under_tower_atk(self, hero, vector_feature, feature_name):
        vector_feature.append(self._is_hero_under_tower_atk(hero))

    def get_enemy_dist(self, hero, vector_feature, feature_name):
        main_hero = self.main_hero_info
        dist = math.sqrt(
            (hero["location"]["x"] - main_hero["location"]["x"]) ** 2
            + (hero["location"]["z"] - main_hero["location"]["z"]) ** 2
        )
        vector_feature.append(dist)
