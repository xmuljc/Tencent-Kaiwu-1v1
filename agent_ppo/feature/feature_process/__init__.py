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
from agent_ppo.feature.feature_process.river_spirit_process import RiverSpiritProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_state_process = GameStateProcess(camp)
        self.river_spirit_process = RiverSpiritProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_state_process = GameStateProcess(camp)
        self.river_spirit_process = RiverSpiritProcess(camp)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.hero_process.process_vec_hero(frame_state)
        enemy_camp_hero_vector_feature = self.hero_process.process_vec_enemy_hero(frame_state)
        organ_feature = self.organ_process.process_vec_organ(frame_state)
        main_hero = self.hero_process.main_hero_info
        enemy_hero = next(iter(self.hero_process.enemy_camp_hero_dict.values()), None)
        soldier_feature = self.soldier_process.process_vec_soldier(frame_state, main_hero, enemy_hero)
        game_state_feature = self.game_state_process.process_vec_game_state(frame_state)
        river_spirit_feature = self.river_spirit_process.process_vec_river_spirit(frame_state, main_hero)

        feature = (
            main_camp_hero_vector_feature
            + enemy_camp_hero_vector_feature
            + organ_feature
            + soldier_feature
            + game_state_feature
            + river_spirit_feature
        )
        return feature
