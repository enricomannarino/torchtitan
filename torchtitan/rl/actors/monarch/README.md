# Monarch actors

This package runs the standalone RL services on
[Monarch](https://github.com/meta-pytorch/monarch). `MonarchActorSystem`
implements the framework-agnostic `ActorSystem` (`rl/actors/system.py`), which
the controller builds from `Controller.Config.actor_system` (default
`MonarchActorSystem.Config()`) and spawns every service through.

Each adapter is a thin Monarch `Actor` subclass of a service with no Monarch
dependency of its own: it declares the service's methods as
`@concurrent_endpoint`s and supplies Monarch-specific constructor arguments
(rank, actor name, `Channel.open`). Domain behavior stays in the service.

| Module | Purpose |
| --- | --- |
| `system.py` | `MonarchActorSystem`: provisions the proc meshes and torch-elastic setup, spawns each service's actors on them, initializes TorchStore on the trainer mesh, and stops the meshes. |
| `trainer.py` | `TrainerActor`, the endpoints of `rl/trainer/trainer.py`'s `Trainer`. |
| `generator.py` | `GeneratorActor`, the endpoints of `rl/generator/generator.py`'s `Generator`; supplies `Channel.open` for the data-parallel result fan-in. |
| `router.py` | `InterGeneratorRouterActor`, the endpoints of `rl/routing/inter_generator.py`'s `InterGeneratorRouter`; enforces that the router is a singleton. |
| `rollout_worker.py` | `RolloutWorkerActor`, a rollout worker in the controller-host CPU pool. |
| `provisioning.py` | Proc meshes: the trainer and generator GPU meshes (`spawn_proc_mesh`, `PerHostProvisioner`), CPU procs on the controller host (`spawn_cpu_procs`), and the Monarch client's `shutdown`. |
| `torch_elastic.py` | Sets the torch-elastic env vars (`RANK`, `MASTER_PORT`, ...) on a proc mesh before `init_process_group`; rank 0 keeps its rendezvous store port open so no other socket takes it. |

## Outside this package

No module outside this package imports Monarch or this package, except the
controller's default for `Controller.Config.actor_system`, which imports
`MonarchActorSystem` lazily. The controller, the rollouter and `train.py` use
the abstract `ActorSystem`.

What remains outside is the call convention on the handles an actor system
returns (see `ActorSystem`): `.call()` with `.get(rank)` on its result,
`.call_one()`, `.choose()` and `.stream()`. The trainer and each generator come
back as a `ServiceHandle` with an all-ranks and a rank-0 handle, so callers never
slice meshes themselves. These calls appear in the controller, the rollouter
(`.choose()` on the worker pool) and the router (on generator handles).

Another actor framework (e.g. Ray) would add its own `ActorSystem`
implementation and adapters, wrapping its handles to provide this call
convention, and be selected with `Controller.Config.actor_system`.
