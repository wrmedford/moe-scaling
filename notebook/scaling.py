import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import math


@dataclass
class MoEConfig:
    """MoE model configuration parameters"""

    num_experts: int
    expert_size: int  # bytes
    experts_per_forward: int


class GPUSystemConfig:
    def __init__(
        self,
        gpu_mem_size: float,
        gpu_bus_bw: float,
        gpu_bandwidth: float,
        gpu_hit_latency: float,
        gpu_miss_latency: float,
    ):
        """
        CPU parameters:
            cpu_cores: the number of cores on the CPU. For example, on a Intel
                       Xeon 6700 with PCIe 4.0, cores may range from 64-144.
                       On the GH200 NVL2, this is 144.

            cpu_l1: L1 cache per core (bytes)
            cpu_l2: L2 cache per core (bytes)
            cpu_l3: Shared L3 cache (bytes)
            cpu_mem_bw: DDR5 memory bandwidth (bytes/sec)

        GPU parameters (Hopper):
            gpu_mem_size: HBM3(e) capacity (bytes)
            gpu_mem_bw: HBM bandwidth (bytes/sec)

        Interconnect:
            interconnect_bw: NVLink/PCIe bandwidth (bytes/sec)
        """

        # GPU parameters
        self.gpu_mem_size = gpu_mem_size
        self.gpu_bus_bw = gpu_bus_bw
        self.gpu_bandwidth = gpu_bandwidth
        self.gpu_hit_latency = gpu_hit_latency
        self.gpu_miss_latency = gpu_miss_latency

        self._calculate_scaling_constants()

    def _calculate_scaling_constants(self):
        """Calculate α and β using memory hierarchy characteristics"""
        # Alpha based on latency ratio between DRAM and cache

        self.alpha = math.log2(self.gpu_miss_latency / self.gpu_hit_latency)

        # Beta based on bandwidth ratios across hierarchy
        self.beta = math.log(self.gpu_bus_bw) / math.log(self.gpu_bandwidth)

    def calculate_cache_probability(self, moe_config: MoEConfig) -> float:
        """
        Calculate probability of expert being in cache using LRU approximation

        Based on the Che approximation for LRU caches.
        Ref: "A unified approach to the performance analysis of caching systems" - Martina et al. 2016

        Still, some assumptions are made:

        * Assumes an LRU cache placement policy
        * Full cache usage, with no on-system, non-model needs, is assumed.
        * Temporal locality is derived.
        * Cache arch is assumed to be "set associative" and it's assumed cache
          blocks of via 8 way association.
        * Cold start mitigation is assumed / simulated
        """
        # Fundamental constraints: return early if cache is empty or expert is no bytes.
        if moe_config.expert_size == 0 or self.gpu_mem_size == 0:
            return 0.0

        # Calculate fundamental cache parameters for given expert size
        num_cache_blocks = self.gpu_mem_size // moe_config.expert_size

        # Typical modern cache associativity
        associativity = 8

        # The number of cache sets for given associativity
        num_sets = num_cache_blocks // associativity

        if num_sets == 0:
            return 0.0  # Expert is too large for cache and cannot fit within blocks.

        # Calculate reuse distance probability (simplified model)
        lambda_param = moe_config.num_experts / (num_sets * associativity)
        p_hit = 1 - math.exp(-lambda_param)

        # Temporal locality correction factor (empirically derived)
        alpha = 0.15 * math.log(self.gpu_bandwidth / 1e9)

        # Final probability with cold start mitigation
        return 1 - min(0.99, max(0.01, p_hit * (1 + alpha)))

    def calculate_forward_experts_miss_probability(
        self, moe_config: MoEConfig
    ) -> float:
        """Calculate cache miss probability"""
        p_cached = self.calculate_cache_probability(moe_config)
        return (1 - p_cached) ** moe_config.experts_per_forward

    def calculate_miss_penalty(self, moe_config: MoEConfig) -> float:
        """Calculate effective miss penalty"""
        p_miss = self.calculate_forward_experts_miss_probability(moe_config)
        return p_miss * (moe_config.expert_size / self.gpu_bus_bw)

    def calculate_performance_gain(self, moe_config: MoEConfig) -> float:
        """Calculate theoretical performance gain based on scaling laws"""
        return moe_config.num_experts**self.alpha / (moe_config.expert_size**self.beta)

    def calculate_effective_bandwidth(self, moe_config: MoEConfig) -> float:
        """Calculate effective bandwidth utilization"""
        p_miss = self.calculate_forward_experts_miss_probability(moe_config)
        p_hit = 1 - p_miss
        return self.gpu_bus_bw * p_miss + self.gpu_bandwidth * p_hit

    def calculate_throughput(
        self, moe_config: MoEConfig, processing_time: float
    ) -> float:
        """Calculate theoretical throughput"""
        miss_penalty = self.calculate_miss_penalty(moe_config)
        return moe_config.experts_per_forward / (processing_time + miss_penalty)


class UnifiedMemSystemConfig:
    """
    Represents a unified memory configuration.
    Basd on the GH200 NVL2

    Grace Hopper Unified Memory Model with Hierarchical Caches.

    Based on NVIDIA GH200 White Paper and ACM SIGARCH 2023 analysis
    """

    def __init__(
        self,
        # 144 ARM Neoverse V2 cores
        cores=144,
        # 1MB L2 cache per core
        cache_size=1024**2,
        # 900GB/s NVLink-C2C
        bus_bandwidth=900 * 1024**3,
        # 9.8TB/s HBM3
        cache_bandwidth=9.8 * 1024**4,
        # 30ns L2 hit
        cache_hit_latency=30e-9,
        # 100ns unified memory access
        cache_miss_latency=100e-9,
    ):
        self.cache_size = cache_size
        self.bus_bandwidth = bus_bandwidth
        self.cache_bandwidth = cache_bandwidth
        self.cache_hit_latency = cache_hit_latency
        self.cache_miss_latency = cache_miss_latency

        # L2 associativity
        self.associativity = 16

        # GH200-specific parameters

        # Streaming Multiprocessors
        self.num_sms = cores

        self.l1_config = {
            "size_per_sm": 128 * 1024,  # 128KB L1/SM (64KB i cache + 64KB d cache)
            "associativity": 4,
            "bandwidth": 80 * 1024**3,  # 80GB/s L1 bandwidth
        }

        self.l2_config = {
            "size": 1 * 1024**2 * self.num_sms,  # 1MB L2 per core
            "associativity": 16,
            "bandwidth": 2.3 * 1024**4,  # 2.3TB/s L2 bandwidth
        }

        self.hbm_config = {
            "size": 288 * 1024**3,  # 288 GB HBM3
            "bandwidth": 9.8 * 1024**4,  # 3.35TB/s
        }

        self.numa_config = {
            "numa_nodes": 4,
            "cross_node_penalty": 0.25,  # 25% bandwidth reduction
        }

        # Calculate α and β scaling constants based on system parameters
        self.alpha = math.log2(self.cache_miss_latency / self.cache_hit_latency)
        self.beta = math.log(self.bus_bandwidth) / math.log(self.cache_bandwidth)

    def _calculate_l1_probability(self, moe_config: MoEConfig) -> float:
        """L1 cache hit probability per SM"""
        l1_blocks = self.l1_config["size_per_sm"] // moe_config.expert_size
        l1_sets = l1_blocks // self.l1_config["associativity"]
        if l1_sets == 0:
            return 0.0

        lambda_l1 = moe_config.num_experts / (l1_sets * self.num_sms)
        return 1 - math.exp(-lambda_l1)

    def _calculate_l2_probability(self, moe_config: MoEConfig) -> float:
        """L2 cache hit probability (shared across SMs)"""
        l2_blocks = self.l2_config["size"] // moe_config.expert_size
        l2_sets = l2_blocks // self.l2_config["associativity"]
        if l2_sets == 0:
            return 0.0

        lambda_l2 = moe_config.num_experts / l2_sets
        return 1 - math.exp(-lambda_l2)

    def _calculate_hbm_probability(self, moe_config: MoEConfig) -> float:
        """HBM hit probability (unified memory model)"""
        hbm_blocks = self.hbm_config["size"] // moe_config.expert_size
        return min(1.0, hbm_blocks / moe_config.num_experts)

    def _calculate_cache_probability(self, moe_config: MoEConfig) -> float:
        """
        Hierarchical cache probability model for GH200

        Combines L1 (per-SM), L2, and HBM using ARM-style
        cache inclusion properties
        """
        p_l1 = self._calculate_l1_probability(moe_config)
        p_l2 = self._calculate_l2_probability(moe_config)
        p_hbm = self._calculate_hbm_probability(moe_config)

        # NUMA-aware unification factor
        numa_factor = 1 - (
            self.numa_config["cross_node_penalty"]
            * (1 - 1 / self.numa_config["numa_nodes"])
        )

        # Combined probability using cache hierarchy inclusion
        return min(0.99, p_l1 + (1 - p_l1) * (p_l2 + (1 - p_l2) * p_hbm * numa_factor))

    def _effective_bandwidth(self, moe_config: MoEConfig) -> float:
        """
        Unified memory bandwidth model with NVLink awareness

        Combines L1/L2/HBM bandwidths with NUMA penalties
        """
        # Base bandwidth components
        bw_l1 = self.l1_config["bandwidth"] * self.num_sms
        bw_l2 = self.l2_config["bandwidth"]
        bw_hbm = self.hbm_config["bandwidth"]

        # Probability weights
        p_l1 = self._calculate_l1_probability(moe_config)
        p_l2 = self._calculate_l2_probability(moe_config)
        p_hbm = self._calculate_hbm_probability(moe_config)

        # NUMA penalty model
        numa_penalty = 1 - (self.numa_config["cross_node_penalty"] * (1 - p_l1**0.5))

        # Effective bandwidth calculation
        return (
            bw_l1 * p_l1
            + bw_l2 * p_l2 * (1 - p_l1)
            + bw_hbm * p_hbm * (1 - p_l1) * (1 - p_l2)
        ) * numa_penalty

    def _unified_miss_penalty(self, moe_config: MoEConfig) -> float:
        """Combined miss penalty accounting for cache hierarchy"""
        t_l1 = 5e-9  # 5ns L1 access
        t_l2 = 30e-9  # 30ns L2 access
        t_hbm = 100e-9  # 100ns HBM access

        p_l1 = self._calculate_l1_probability(moe_config)
        p_l2 = self._calculate_l2_probability(moe_config)
        p_hbm = self._calculate_hbm_probability(moe_config)

        return (
            t_l1 * p_l1
            + t_l2 * p_l2 * (1 - p_l1)
            + t_hbm * p_hbm * (1 - p_l1) * (1 - p_l2)
        )

    def calculate_cache_probability(self, moe_config: MoEConfig) -> float:
        """Calculate probability of experts being in cache"""
        return self._calculate_cache_probability(moe_config)

    def calculate_forward_experts_miss_probability(
        self, moe_config: MoEConfig
    ) -> float:
        """Calculate cache miss probability"""
        p_cached = self.calculate_cache_probability(moe_config)
        return (1 - p_cached) ** moe_config.experts_per_forward

    def calculate_miss_penalty(self, moe_config: MoEConfig) -> float:
        """Calculate effective miss penalty"""
        return self._unified_miss_penalty(moe_config)

    def calculate_performance_gain(self, moe_config: MoEConfig) -> float:
        """Calculate theoretical performance gain based on scaling laws"""
        return moe_config.num_experts**self.alpha / (moe_config.expert_size**self.beta)

    def calculate_effective_bandwidth(self, moe_config: MoEConfig) -> float:
        """Calculate effective bandwidth utilization"""
        return self._effective_bandwidth(moe_config)

    def calculate_throughput(
        self, moe_config: MoEConfig, processing_time: float
    ) -> float:
        """Calculate theoretical throughput"""
        miss_penalty = self.calculate_miss_penalty(moe_config)
        return moe_config.experts_per_forward / (processing_time + miss_penalty)


class ScalingLawSimulator:
    """Simulator for MoE scaling law experiments"""

    def __init__(self, system_config: GPUSystemConfig):
        self.system = system_config

    def calculate_cache_probability(self, moe_config: MoEConfig) -> float:
        return self.system.calculate_cache_probability(moe_config)

    def calculate_miss_probability(self, moe_config: MoEConfig) -> float:
        return self.system.calculate_forward_experts_miss_probability(moe_config)

    def calculate_miss_penalty(self, moe_config: MoEConfig) -> float:
        return self.system.calculate_miss_penalty(moe_config)

    def calculate_performance_gain(self, moe_config: MoEConfig) -> float:
        return self.system.calculate_performance_gain(moe_config)

    def calculate_effective_bandwidth(self, moe_config: MoEConfig) -> float:
        return self.system.calculate_effective_bandwidth(moe_config)

    def calculate_throughput(
        self, moe_config: MoEConfig, processing_time: float
    ) -> float:
        return self.system.calculate_throughput(moe_config, processing_time)


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
                    "single_expert_cache_prob": self.simulator.calculate_cache_probability(
                        config
                    ),
                    "forward_experts_miss_prob": self.simulator.calculate_miss_probability(
                        config
                    ),
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
                    "single_expert_cache_prob": self.simulator.calculate_cache_probability(
                        config
                    ),
                    "forward_experts_miss_prob": self.simulator.calculate_miss_probability(
                        config
                    ),
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
        df: pd.DataFrame,
        x_col: str,
        metrics: List[str],
        title: str,
        is_log: bool = False,
    ):
        """Plot scaling relationships for multiple metrics"""
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(df[x_col], df[metric], label=metric)
        plt.xlabel(x_col)
        plt.ylabel("Normalized Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if is_log:
            plt.xscale("log")
            plt.yscale("log")

        plt.show()

    @staticmethod
    def plot_scaling_comparison(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        a_label: str,
        b_label: str,
        x_col: str,
        metrics: List[str],
        title: str,
        is_log: bool = False,
    ):
        """Compare two data frames"""
        plt.figure(figsize=(14, 10))
        for metric in metrics:
            plt.plot(df_a[x_col], df_a[metric], label=f"{a_label} {metric}")
            plt.plot(df_b[x_col], df_b[metric], label=f"{b_label} {metric}")

        plt.xlabel(x_col)
        plt.ylabel("Normalized Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if is_log:
            plt.xscale("log")
            plt.yscale("log")

        plt.tight_layout()
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
