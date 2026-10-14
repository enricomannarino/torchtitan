# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The actor framework interface the RL controller places its services through."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.rl.generator.generator import Generator
    from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
    from torchtitan.rl.rollout.rollouter import RolloutWorker
    from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
    from torchtitan.rl.trainer.trainer import Trainer


@dataclass(frozen=True, slots=True)
class ServiceHandle:
    """Handles to the actors of one SPMD service: the trainer, or one generator."""

    all_ranks: Any
    """Endpoint calls on this reach every rank, e.g. ``all_ranks.push_model_state_dict.call()``."""

    rank0: Any
    """Endpoint calls on this reach rank 0 only, e.g. ``rank0.generate.call_one(...)``."""


class ActorSystem(Configurable, abc.ABC):
    """Places the RL services on processes and returns handles to them.

    The services (``Trainer``, ``Generator``, ``InterGeneratorRouter``,
    ``RolloutWorker``) are framework-free classes; an actor system runs each in
    its framework's actors. The controller builds it from
    ``Controller.Config.actor_system`` and drives it: ``provision`` once, then a
    ``spawn_*`` call per service, and ``stop`` on close. The rollouter spawns and
    stops its worker pool through it.

    Handles follow one call convention. An endpoint ``handle.<method>`` is called
    with:

    - ``.call(...)``: on every actor of the handle; the result holds each rank's
      value, read with ``.get(rank)``;
    - ``.call_one(...)``: on a single-actor handle; returns its value;
    - ``.choose(...)``: on any one actor of the handle;
    - ``.stream(...)``: on every actor, yielding each rank's value as it arrives.

    These are Monarch's endpoint calls; another framework's actor system wraps its
    handles to provide them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abc.abstractmethod
    async def provision(
        self,
        *,
        trainer_parallelism: ParallelismConfig,
        generator_parallelism: InferenceParallelismConfig,
        num_generators: int,
    ) -> None:
        """Reserve the processes for the trainer, ``num_generators`` generators and
        the generator router, before any of them is spawned."""

    @abc.abstractmethod
    async def spawn_trainer(
        self, config: Trainer.Config, **kwargs: Any
    ) -> ServiceHandle:
        """Spawn the trainer; ``kwargs`` go to ``Trainer``'s constructor."""

    @abc.abstractmethod
    async def spawn_generators(
        self, config: Generator.Config, **kwargs: Any
    ) -> list[ServiceHandle]:
        """Spawn the provisioned generators; ``kwargs`` go to ``Generator``'s
        constructor."""

    @abc.abstractmethod
    async def spawn_generator_router(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generators: list[ServiceHandle],
    ) -> Any:
        """Spawn the singleton router over ``generators``; returns a single-actor
        handle."""

    @abc.abstractmethod
    async def spawn_rollout_workers(
        self, config: RolloutWorker.Config, *, num_workers: int, num_threads: int
    ) -> Any:
        """Spawn a pool of ``num_workers`` rollout worker processes on the
        controller host, each running ``num_threads`` rollout threads."""

    @abc.abstractmethod
    async def stop_rollout_workers(self) -> None:
        """Stop the rollout worker pool, if any."""

    @abc.abstractmethod
    async def initialize_torchstore(self) -> None:
        """Initialize TorchStore, with its storage colocated with the trainer.

        TODO: move to a model state dict transport, which owns its placement.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop every process ``provision`` reserved, best-effort."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release the framework's client in this process, after the event loop
        has finished."""
