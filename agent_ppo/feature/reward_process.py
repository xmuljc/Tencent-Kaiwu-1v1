#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    DEFENSE_ORGAN_SUBTYPES = {21, 24}
    TOWER_SUB_TYPE = 21
    SOLDIER_OR_MONSTER_ACTOR_TYPE = 1

    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    def _safe_ratio(self, value, max_value):
        if max_value <= 0:
            return 0.0
        return 1.0 * value / max_value

    def _find_hero_by_camp(self, frame_data, camp):
        for hero in frame_data.get("hero_states", []):
            if hero.get("camp") == camp:
                return hero
        return None

    def _find_tower_by_camp(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") == camp and organ.get("sub_type") == self.TOWER_SUB_TYPE:
                return organ
        return None

    def _find_enemy_tower(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") != camp and organ.get("sub_type") == self.TOWER_SUB_TYPE:
                return organ
        return None

    def _defense_organ_hp_rate(self, frame_data, camp):
        total_hp, total_max_hp = 0.0, 0.0
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") != camp or organ.get("sub_type") not in self.DEFENSE_ORGAN_SUBTYPES:
                continue
            total_hp += organ.get("hp", 0)
            total_max_hp += organ.get("max_hp", 0)
        return self._safe_ratio(total_hp, total_max_hp)

    def _total_exp(self, hero):
        level = hero.get("level", 1)
        current_exp = hero.get("exp", 0)
        completed_level_exp = 0
        for level_idx in range(1, level):
            completed_level_exp += self.m_each_level_max_exp.get(level_idx, 0)
        return completed_level_exp + current_exp

    def _last_hit_count(self, frame_data, camp, hero):
        if hero is None:
            return 0

        count = 0
        hero_runtime_id = hero.get("runtime_id")
        dead_actions = frame_data.get("frame_action", {}).get("dead_action", [])
        for dead_action in dead_actions:
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            death_camp = death.get("camp")

            is_enemy_unit = (
                death.get("actor_type") == self.SOLDIER_OR_MONSTER_ACTOR_TYPE
                and death_camp not in (None, 0, camp)
            )
            killed_by_hero = killer.get("runtime_id") == hero_runtime_id
            if is_enemy_unit and killed_by_hero:
                count += 1
        return count

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        main_hero = self._find_hero_by_camp(frame_data, camp)
        main_tower = self._find_tower_by_camp(frame_data, camp)
        enemy_tower = self._find_enemy_tower(frame_data, camp)

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if main_hero is None:
                reward_struct.cur_frame_value = 0.0
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = self._safe_ratio(
                    main_hero.get("hp", 0), main_hero.get("max_hp", 0)
                )
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = self._defense_organ_hp_rate(frame_data, camp)
            elif reward_name == "money":
                reward_struct.cur_frame_value = main_hero.get("money_cnt", main_hero.get("money", 0))
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self._safe_ratio(
                    main_hero.get("ep", 0), main_hero.get("max_ep", 0)
                )
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero.get("dead_cnt", 0)
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero.get("kill_cnt", 0)
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self._total_exp(main_hero)
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value += self._last_hit_count(frame_data, camp, main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0

        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        hero_hp_rate = self._safe_ratio(main_hero.get("hp", 0), main_hero.get("max_hp", 0))
        if hero_hp_rate > 0.99 and dist_main2emy > 0 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            # Calculate zero-sum reward
            # 计算零和奖励
            reward_struct.cur_frame_value = (
                self.m_main_calc_frame_map[reward_name].cur_frame_value
                - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
            )
            reward_struct.last_frame_value = (
                self.m_main_calc_frame_map[reward_name].last_frame_value
                - self.m_enemy_calc_frame_map[reward_name].last_frame_value
            )
            reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
