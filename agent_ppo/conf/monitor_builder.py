#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="PPO Ratio",
            name_en="ppo_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="ratio_mean",
            expr="round(avg(ratio_mean{}), 0.0001)",
        )
        .add_metric(
            metrics_name="ratio_max",
            expr="round(avg(ratio_max{}), 0.0001)",
        )
        .end_panel()
        .add_panel(
            name="Advantage",
            name_en="advantage",
            type="line",
        )
        .add_metric(
            metrics_name="advantage_mean",
            expr="round(avg(advantage_mean{}), 0.0001)",
        )
        .add_metric(
            metrics_name="advantage_abs_mean",
            expr="round(avg(advantage_abs_mean{}), 0.0001)",
        )
        .add_metric(
            metrics_name="negative_advantage_ratio",
            expr="round(avg(negative_advantage_ratio{}), 0.0001)",
        )
        .end_panel()
        .add_panel(
            name="Dual Clip",
            name_en="dual_clip",
            type="line",
        )
        .add_metric(
            metrics_name="dual_clip_active_ratio",
            expr="round(avg(dual_clip_active_ratio{}), 0.0001)",
        )
        .end_panel()
        .add_panel(
            name="Value Quality",
            name_en="value_quality",
            type="line",
        )
        .add_metric(
            metrics_name="return_mean",
            expr="round(avg(return_mean{}), 0.0001)",
        )
        .add_metric(
            metrics_name="value_mean",
            expr="round(avg(value_mean{}), 0.0001)",
        )
        .add_metric(
            metrics_name="value_error_abs",
            expr="round(avg(value_error_abs{}), 0.0001)",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="Reward Diagnostics",
            group_name_en="reward_diagnostics",
        )
        .add_panel(
            name="Reward Components",
            name_en="reward_components",
            type="line",
        )
        .add_metric(
            metrics_name="reward_tower_hp_point",
            expr="round(avg(reward_tower_hp_point{}), 0.0001)",
        )
        .add_metric(
            metrics_name="reward_money",
            expr="round(avg(reward_money{}), 0.0001)",
        )
        .add_metric(
            metrics_name="reward_exp",
            expr="round(avg(reward_exp{}), 0.0001)",
        )
        .add_metric(
            metrics_name="reward_hp_advantage",
            expr="round(avg(reward_hp_advantage{}), 0.0001)",
        )
        .add_metric(
            metrics_name="reward_kill_death",
            expr="round(avg(reward_kill_death{}), 0.0001)",
        )
        .add_metric(
            metrics_name="reward_win",
            expr="round(avg(reward_win{}), 0.0001)",
        )
        .end_panel()
        .add_panel(
            name="Reward Stage",
            name_en="reward_stage",
            type="line",
        )
        .add_metric(
            metrics_name="reward_stage_progress",
            expr="round(avg(reward_stage_progress{}), 0.0001)",
        )
        .add_metric(
            metrics_name="stage_total_frames",
            expr="round(avg(stage_total_frames{}), 0.01)",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="Episode Quality",
            group_name_en="episode_quality",
        )
        .add_panel(
            name="Episode Frame",
            name_en="episode_frame",
            type="line",
        )
        .add_metric(
            metrics_name="episode_frame",
            expr="round(avg(episode_frame{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="Episode Reward",
            name_en="episode_reward",
            type="line",
        )
        .add_metric(
            metrics_name="episode_reward_train",
            expr="round(avg(episode_reward_train{}), 0.01)",
        )
        .add_metric(
            metrics_name="episode_reward_eval",
            expr="round(avg(episode_reward_eval{}), 0.01)",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
