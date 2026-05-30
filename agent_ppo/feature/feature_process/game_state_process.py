#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os


class GameStateProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.get_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.feature_config)
        self.one_unit_feature_num = 8
        self.unit_buff_num = 1

    def get_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "game_state_feature_config.ini")
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

    def process_vec_game_state(self, frame_state):
        main_hero = None
        enemy_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == self.main_camp:
                main_hero = hero
            else:
                enemy_hero = hero

        self._frame_no = frame_state.get("frame_no", 0)
        self._main_hero = main_hero
        self._enemy_hero = enemy_hero

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

    def get_g_game_time(self, value):
        value.append(min(int(self._frame_no) // 1800, 4))

    def get_hp_advantage(self, value):
        if self._main_hero and self._enemy_hero:
            main_hp = self._main_hero["hp"] / self._main_hero["max_hp"] if self._main_hero.get("max_hp", 0) > 0 else 0
            enemy_hp = self._enemy_hero["hp"] / self._enemy_hero["max_hp"] if self._enemy_hero.get("max_hp", 0) > 0 else 0
            value.append(main_hp - enemy_hp)
        else:
            value.append(0.0)

    def get_level_advantage(self, value):
        if self._main_hero and self._enemy_hero:
            value.append(self._main_hero.get("level", 1) - self._enemy_hero.get("level", 1))
        else:
            value.append(0)

    def get_money_advantage(self, value):
        if self._main_hero and self._enemy_hero:
            value.append(self._main_hero.get("money", 0) - self._enemy_hero.get("money", 0))
        else:
            value.append(0)
