# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch actor system: the RL services as Monarch actors on this host."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torchstore as ts
from monarch.actor import ProcMesh

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl.actors.monarch.generator import GeneratorActor
from torchtitan.rl.actors.monarch.provisioning import (
    compute_generator_world_size,
    compute_trainer_world_size,
    shutdown,
    spawn_cpu_procs,
    spawn_proc_mesh,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.router import InterGeneratorRouterActor
from torchtitan.rl.actors.monarch.torch_elastic import setup_torch_elastic_env
from torchtitan.rl.actors.monarch.trainer import TrainerActor
from torchtitan.rl.actors.system import ActorSystem, ServiceHandle
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


class MonarchActorSystem(ActorSystem):
    """Runs the RL services as Monarch actors on proc meshes of this host.

    The trainer and each generator get their own GPU proc mesh, partitioned with
    ``CUDA_VISIBLE_DEVICES``; the router and the rollout worker pool get CPU procs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActorSystem.Config):
        pass

    def __init__(self, config: Config):
        self.config = config
        self._trainer_mesh: ProcMesh | None = None
        self._generator_meshes: list[ProcMesh] = []
        self._router_mesh: ProcMesh | None = None
        self._rollout_worker_mesh: ProcMesh | None = None

    async def provision(
        self,
        *,
        trainer_parallelism: ParallelismConfig,
        generator_parallelism: InferenceParallelismConfig,
        num_generators: int,
    ) -> None:
        self._trainer_mesh, self._generator_meshes = spawn_proc_mesh(
            compute_trainer_world_size(trainer_parallelism),
            compute_generator_world_size(generator_parallelism),
            host_meshes=None,
            num_generators=num_generators,
        )
        # One process, so the router is a singleton and every caller reaches
        # it with `call_one`. It gets its own mesh rather than sharing the
        # controller's process so routing does not contend with the training
        # loop for the controller's GIL.
        self._router_mesh = spawn_cpu_procs(1)

        await setup_torch_elastic_env(self._trainer_mesh)
        for generator_mesh in self._generator_meshes:
            await setup_torch_elastic_env(generator_mesh)

    async def spawn_trainer(
        self, config: Trainer.Config, **kwargs: Any
    ) -> ServiceHandle:
        assert self._trainer_mesh is not None, "provision before spawning"
        trainer = self._trainer_mesh.spawn("trainer", TrainerActor, config, **kwargs)
        return ServiceHandle(
            all_ranks=trainer, rank0=trainer.flatten("rank").slice(rank=0)
        )

    async def spawn_generators(
        self, config: Generator.Config, **kwargs: Any
    ) -> list[ServiceHandle]:
        generators = []
        for idx, generator_mesh in enumerate(self._generator_meshes):
            actor_name = (
                "generator" if len(self._generator_meshes) == 1 else f"generator_{idx}"
            )
            generator = generator_mesh.spawn(
                actor_name, GeneratorActor, config, **kwargs
            )
            generators.append(
                ServiceHandle(
                    all_ranks=generator,
                    rank0=generator.flatten("rank").slice(rank=0),
                )
            )
        return generators

    async def spawn_generator_router(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generators: list[ServiceHandle],
    ) -> Any:
        assert self._router_mesh is not None, "provision before spawning"
        return self._router_mesh.spawn(
            "generator_router",
            InterGeneratorRouterActor,
            config,
            generators=generators,
        )

    async def spawn_rollout_workers(
        self, config: RolloutWorker.Config, *, num_workers: int, num_threads: int
    ) -> Any:
        self._rollout_worker_mesh = spawn_cpu_procs(num_workers)
        return self._rollout_worker_mesh.spawn(
            "rollout_worker",
            RolloutWorkerActor,
            worker_config=config,
            num_threads=num_threads,
        )

    async def stop_rollout_workers(self) -> None:
        worker_mesh = self._rollout_worker_mesh
        self._rollout_worker_mesh = None
        if worker_mesh is not None:
            await worker_mesh.stop()

    async def initialize_torchstore(self) -> None:
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(
            mesh=self._trainer_mesh,
            strategy=ts.TorchStoreStrategy(),
            client_type=ts.ClientType.ROUTING,
        )

    async def stop(self) -> None:
        meshes = [
            mesh
            for mesh in (self._router_mesh, self._trainer_mesh, *self._generator_meshes)
            if mesh is not None
        ]
        self._router_mesh, self._trainer_mesh, self._generator_meshes = None, None, []
        for i, mesh in enumerate(meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)

    def shutdown(self) -> None:
        shutdown()
