#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess
from agent_ppo.feature.feature_process.game_state_process import GameStateProcess
from agent_ppo.feature.feature_process.debug_feature_process import DebugFeatureProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_state_process = GameStateProcess(camp)
        self.debug_feature_process = DebugFeatureProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_state_process = GameStateProcess(camp)
        self.debug_feature_process = DebugFeatureProcess(camp)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.hero_process.process_vec_hero(frame_state)
        enemy_camp_hero_vector_feature = self.hero_process.process_vec_enemy_hero(frame_state)
        organ_feature = self.organ_process.process_vec_organ(frame_state)
        main_hero = self.hero_process.main_hero_info
        soldier_feature = self.soldier_process.process_vec_soldier(frame_state, main_hero)
        game_state_feature = self.game_state_process.process_vec_game_state(frame_state)
        enemy_hero = next(iter(self.hero_process.enemy_camp_hero_dict.values()), None)
        debug_feature = self.debug_feature_process.process_vec_debug(frame_state, main_hero, enemy_hero)

        feature = (
            main_camp_hero_vector_feature
            + enemy_camp_hero_vector_feature
            + organ_feature
            + soldier_feature
            + game_state_feature
            + debug_feature
        )
        return feature
