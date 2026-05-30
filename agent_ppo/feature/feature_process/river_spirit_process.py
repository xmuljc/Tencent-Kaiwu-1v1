#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math


RIVER_SPIRIT_CONFIG_ID = 6827
RIVER_SPIRIT_FEATURE_DIM = 8


class RiverSpiritProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == 2 or camp == "PLAYERCAMP_2"

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, value))

    @staticmethod
    def _distance_xz(loc_a, loc_b):
        return math.sqrt(
            (loc_a.get("x", 0) - loc_b.get("x", 0)) ** 2
            + (loc_a.get("z", 0) - loc_b.get("z", 0)) ** 2
        )

    def _is_river_spirit(self, npc):
        return (
            npc.get("config_id") == RIVER_SPIRIT_CONFIG_ID
            and npc.get("actor_type") == 1
            and npc.get("sub_type") == 0
        )

    def _select_river_spirit(self, frame_state):
        for npc in frame_state.get("npc_states", []):
            if self._is_river_spirit(npc):
                return npc
        return None

    def _relative_axis(self, river_spirit, main_hero, axis):
        diff = river_spirit.get("location", {}).get(axis, 0) - main_hero.get("location", {}).get(axis, 0)
        if self.transform_camp2_to_camp1:
            diff = -diff
        return self._clip01((diff + 25000.0) / 50000.0)

    def process_vec_river_spirit(self, frame_state, main_hero):
        river_spirit = self._select_river_spirit(frame_state)
        if river_spirit is None or main_hero is None:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        hp = river_spirit.get("hp", 0)
        max_hp = river_spirit.get("max_hp", 0)
        hp_rate = hp / max_hp if max_hp > 0 else 0.0
        distance = self._distance_xz(river_spirit.get("location") or {}, main_hero.get("location") or {})
        return [
            1.0,
            self._relative_axis(river_spirit, main_hero, "x"),
            self._relative_axis(river_spirit, main_hero, "z"),
            self._clip01(distance / 30000.0),
            self._clip01(hp / 6000.0),
            self._clip01(hp_rate),
            self._clip01(max_hp / 6000.0),
            self._clip01(river_spirit.get("kill_income", 0) / 150.0),
        ]
