import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import math


@dataclass
class SystemConfig:
    """
    Hardware system configuration parameters

    Attributes:
        cache_size (bytes): Total cache size
        bus_bandwidth (bytes/sec): CPU <-> Accelerator bandwidth
        cache_bandwidth (bytes/sec): On-chip memory bandwidth
        cache_hit_latency (seconds): Latency for cache hits
        cache_miss_latency (seconds): Penalty for cache misses
    """

    cache_size: int
    bus_bandwidth: float
    cache_bandwidth: float
    cache_hit_latency: float
    cache_miss_latency: float

    def __post_init__(self):
        """
        After initialization, asserts that the provided options are valid.
        """
        assert self.cache_size > 0, "Cache size must be positive"
        assert (
            self.bus_bandwidth < self.cache_bandwidth
        ), "Cache bandwidth should exceed bus bandwidth"


@dataclass
class MoEConfig:
    """MoE model configuration parameters"""

    num_experts: int
    expert_size: int  # bytes
    experts_per_forward: int


class ScalingLawSimulator:
    """Simulator for MoE scaling law experiments"""

    def __init__(self, system_config: SystemConfig):
        self.system = system_config
        self._calculate_scaling_constants()

    def _calculate_scaling_constants(self):
        """Calculate α and β scaling constants based on system parameters"""
        self.alpha = math.log2(
            self.system.cache_miss_latency / self.system.cache_hit_latency
        )
        self.beta = math.log(self.system.bus_bandwidth) / math.log(
            self.system.cache_bandwidth
        )

    def calculate_cache_probability(self, moe_config: MoEConfig) -> float:
        """Calculate probability of experts being in cache"""
        cache_capacity = self.system.cache_size / moe_config.expert_size
        return min(1.0, cache_capacity / moe_config.num_experts)

    def calculate_miss_probability(self, moe_config: MoEConfig) -> float:
        """Calculate cache miss probability"""
        p_cached = self.calculate_cache_probability(moe_config)
        return (1 - p_cached) ** moe_config.experts_per_forward

    def calculate_miss_penalty(self, moe_config: MoEConfig) -> float:
        """Calculate effective miss penalty"""
        p_miss = self.calculate_miss_probability(moe_config)
        return p_miss * (moe_config.expert_size / self.system.bus_bandwidth)

    def calculate_performance_gain(self, moe_config: MoEConfig) -> float:
        """Calculate theoretical performance gain based on scaling laws"""
        return moe_config.num_experts**self.alpha / (moe_config.expert_size**self.beta)

    def calculate_effective_bandwidth(self, moe_config: MoEConfig) -> float:
        """Calculate effective bandwidth utilization"""
        p_miss = self.calculate_miss_probability(moe_config)
        p_hit = 1 - p_miss
        return self.system.bus_bandwidth * p_miss + self.system.cache_bandwidth * p_hit

    def calculate_throughput(
        self, moe_config: MoEConfig, processing_time: float
    ) -> float:
        """Calculate theoretical throughput"""
        miss_penalty = self.calculate_miss_penalty(moe_config)
        return moe_config.experts_per_forward / (processing_time + miss_penalty)


class ExperimentRunner:
    """Runner for MoE scaling experiments"""

    def __init__(self, simulator: ScalingLawSimulator):
        self.simulator = simulator

    def expert_count_sweep(
        self,
        start_experts: int,
        end_experts: int,
        step: int,
        expert_size: int,
        experts_per_forward: int,
    ) -> pd.DataFrame:
        """
        Experiment runner entry-point.

        Runs the full experiment sweep over the number of experts
        from the start of experts size to end of experts size
        """
        results = []
        num_experts = np.logspace(
            np.log10(start_experts), np.log10(end_experts), num=step
        )
        for n in num_experts.astype(int):
            config = MoEConfig(n, expert_size, experts_per_forward)
            results.append(
                {
                    "num_experts": n,
                    "cache_prob": self.simulator.calculate_cache_probability(config),
                    "miss_prob": self.simulator.calculate_miss_probability(config),
                    "perf_gain": self.simulator.calculate_performance_gain(config),
                    "eff_bandwidth": self.simulator.calculate_effective_bandwidth(
                        config
                    ),
                }
            )
        return pd.DataFrame(results)

    def expert_size_sweep(
        self,
        num_experts: int,
        start_size: int,
        end_size: int,
        step: int,
        experts_per_forward: int,
    ) -> pd.DataFrame:
        """Run sweep over expert sizes"""
        results = []
        for expert_size in range(start_size, end_size + 1, step):
            config = MoEConfig(num_experts, expert_size, experts_per_forward)
            results.append(
                {
                    "expert_size": expert_size,
                    "cache_prob": self.simulator.calculate_cache_probability(config),
                    "miss_prob": self.simulator.calculate_miss_probability(config),
                    "perf_gain": self.simulator.calculate_performance_gain(config),
                    "eff_bandwidth": self.simulator.calculate_effective_bandwidth(
                        config
                    ),
                }
            )
        return pd.DataFrame(results)


class Visualizer:
    """Visualization utilities for experiment results"""

    @staticmethod
    def plot_scaling_relationships(
        df: pd.DataFrame, x_col: str, metrics: List[str], title: str
    ):
        """Plot scaling relationships for multiple metrics"""
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(df[x_col], df[metric], label=metric)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(x_col)
        plt.ylabel("Normalized Value (log scale)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_performance_contour(
        num_experts_range: np.ndarray,
        expert_sizes_range: np.ndarray,
        perf_matrix: np.ndarray,
        title: str,
    ):
        """Plot performance contour map"""
        plt.figure(figsize=(10, 8))
        plt.contourf(
            num_experts_range,
            expert_sizes_range,
            perf_matrix,
            levels=20,
            cmap="viridis",
        )
        plt.colorbar(label="Performance Gain")
        plt.xlabel("Number of Experts")
        plt.ylabel("Expert Size (bytes)")
        plt.title(title)
        plt.show()
