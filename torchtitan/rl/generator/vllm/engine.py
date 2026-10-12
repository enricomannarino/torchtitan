# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference engine for the RL generator: every vLLM call the generator makes."""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.exceptions import VLLMClientError
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.rl.generator.vllm.batch_invariance import (
    force_logprobs_fn_for_batch_invariance,
)
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
)
from torchtitan.rl.generator.vllm.stat_logger import (
    StatLoggerContext,
    VllmOtelStatLogger,
)
from torchtitan.tools.utils import has_cuda_capability

if TYPE_CHECKING:
    from torchtitan.rl.generator.generator import VLLMGenerator

logger = logging.getLogger(__name__)


def setup_vllm(config: VLLMGenerator.Config) -> None:
    """Process-wide vLLM setup, before any ``VLLMEngine`` is built: register the TorchTitan
    model and config parser with vLLM, and select vLLM's model runner."""
    register_to_vllm(
        config.model,
        parallelism=config.parallelism,
        checkpointer_config=config.checkpointer,
        dist_moe_runtime=config.dist_moe_runtime,
    )
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if config.debug.batch_invariant:
        # The vLLM v2 logprob Triton kernel bypasses the batch-invariant aten
        # overrides; route it through trainer's function to match the trainer exactly.
        force_logprobs_fn_for_batch_invariance()


def build_engine_args(
    config: VLLMGenerator.Config, *, model_path: str, max_num_seqs: int
) -> EngineArgs:
    """Build the vLLM ``EngineArgs`` for ``config``.

    Args:
        config: The generator config.
        model_path: Path to the HF model checkpoint.
        max_num_seqs: vLLM's upper bound on concurrently scheduled sequences.
    """
    model_config = config.model
    attention_backend = model_config.first_base_attention_backend
    assert isinstance(
        attention_backend,
        (VarlenInnerAttention.Config, FlexInnerAttention.Config),
    ), "Only varlen and flex attention backends are allowed."

    enable_ep = config.parallelism.expert_parallel_degree > 1
    engine_kwargs = dict(
        # ``model`` is the path to the HF checkpoint directory. The
        # config is sourced from TorchTitan's model config via
        # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
        # read), but vLLM still uses this path to locate the
        # tokenizer assets and the safetensors weight shards.
        model=model_path,
        trust_remote_code=True,
        # Use the torchtitan custom config parser (registered by
        # register_to_vllm in setup_vllm). It builds PretrainedConfig from
        # model config instead of reading config.json from disk.
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=config.model_dtype,
        tensor_parallel_size=config.parallelism.tensor_parallel_degree,
        data_parallel_size=config.parallelism.data_parallel_degree,
        # NOTE: Monarch launches the generator workers and sets the torch
        # elastic distributed env; with external_launcher, vLLM uses that
        # world to build its process groups. vLLM does not take an
        # explicit EP degree: when this boolean is set, it converts all
        # DP * TP ranks into the expert-parallel group for MoE layers.
        enable_expert_parallel=enable_ep,
        worker_cls=TORCHTITAN_WORKER_CLS,
        # Monarch already spawned TP workers via proc mesh. "external_launcher"
        # tells vLLM to run one worker per process (no subprocess spawning)
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=config.gpu_memory_limit,
        enforce_eager=config.cuda_graph.mode == "NONE",
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(attention_backend, FlexInnerAttention.Config)
                else AttentionBackendEnum.CUSTOM
            ),
        ),
        # Enables RequestOutput.metrics, so generator metrics can be returned
        disable_log_stats=False,
        enable_cumem_allocator=False,
        # Token-in-token-out: prompts and outputs are token ids, so vLLM
        # needs no tokenizer. This also drops the tokenizer's eos_token_id
        # as a stop; the generation config's eos ids are dropped by
        # ignore_eos in _build_sampling_params.
        skip_tokenizer_init=True,
    )
    engine_kwargs["max_model_len"] = model_config.max_context_length
    # Return logprobs of the distribution vLLM samples from (after temperature). vLLM's default
    # returns the raw model's logprobs, before temperature.
    engine_kwargs["logprobs_mode"] = "processed_logprobs"
    engine_kwargs["max_num_seqs"] = max_num_seqs
    if config.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
    # Continuous batching requires FCFS scheduling: admission order must equal the
    # broadcast order on every rank
    engine_kwargs["scheduling_policy"] = "fcfs"
    # Which sliding-window / Mamba (GDN) state blocks are hashed into the prefix
    # cache; full-attention groups hash every block regardless.
    # - 0 (vLLM default since v0.29): only the replay boundary (the last block
    #   boundary before the prompt's final token) and shared-prefix junctions.
    #   Other prompt blocks and every block crossed during decode are freed
    #   unhashed, so a multi-turn rollout's next turn, whose prompt contains the
    #   previous completion, cannot reuse anything past the previous prompt.
    # - None: every full block boundary, as before v0.29. Blocks are still freed
    #   when the request moves past them; hashed ones stay reusable until evicted.
    engine_kwargs["prefix_cache_retention_interval"] = None
    # FA2 requires block_size to be a multiple of 256
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    vllm_compilation_config = config.cuda_graph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        expert_sequence_parallel_size=config.parallelism.expert_sequence_parallel_size,
        enable_sequence_parallel=config.parallelism.enable_sequence_parallel,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if config.debug.seed is not None:
        engine_kwargs["seed"] = config.debug.seed
    add_extra_vllm_engine_args(engine_kwargs, config.extra_vllm_engine_args)
    return EngineArgs(**engine_kwargs)


def add_extra_vllm_engine_args(
    engine_kwargs: dict[str, Any], extra_vllm_engine_args: dict[str, Any]
) -> None:
    """Add ``VLLMGenerator.Config.extra_vllm_engine_args`` to ``engine_kwargs`` in place.

    Raises ``ValueError`` for keys TorchTitan already set in ``engine_kwargs`` or
    derives from a config field, so they cannot silently diverge from it.
    """
    # EngineArgs that TorchTitan sets only when the mapped VLLMGenerator.Config
    # field is set (or derives from it), so they may be absent from engine_kwargs.
    config_field_by_engine_arg = {
        "max_num_batched_tokens": "max_num_batched_tokens",
        "compilation_config": "cuda_graph",
        "seed": "debug.seed",
    }
    owned_keys = sorted(
        extra_vllm_engine_args.keys()
        & (engine_kwargs.keys() | config_field_by_engine_arg.keys())
    )
    if owned_keys:
        details = ", ".join(
            (
                f"{key} (use VLLMGenerator.Config.{config_field_by_engine_arg[key]})"
                if key in config_field_by_engine_arg
                else key
            )
            for key in owned_keys
        )
        raise ValueError(
            f"extra_vllm_engine_args cannot set EngineArgs that TorchTitan controls: {details}"
        )
    engine_kwargs.update(extra_vllm_engine_args)


def build_stat_loggers(
    config: VllmOtelStatLogger.Config | None, context: StatLoggerContext
) -> list | None:
    """Build the vLLM stat logger factories passed to ``VLLMEngine``, or None without a config."""
    if config is None:
        logger.info(
            "VllmOtelStatLogger inactive because "
            "vllm_stat_logger=None. To record vLLM metrics, set it "
            "to VllmOtelStatLogger.Config() and set "
            "OTEL_METRICS_EXPORTER=jsonl or otlp"
        )
        return None

    def build_stat_logger(vllm_config, engine_index):
        return config.build(
            vllm_config=vllm_config,
            engine_index=engine_index,
            context=context,
        )

    return [build_stat_logger]


class VLLMEngine:
    """The generator's vLLM ``LLMEngine``, behind the calls the engine loop makes.

    ``LLMEngine`` is not thread-safe: build and use this only on the generator's
    engine thread (see ``EngineLoop``).

    Args:
        engine_args: From ``build_engine_args``.
        reset_kv_cache_on_model_state_dict_load: ``VLLMGenerator.Config``'s field.
            When False, each prompt is salted with its request's min policy version,
            so it reuses only KV computed under that version.
        stat_loggers: From ``build_stat_loggers``.
    """

    def __init__(
        self,
        engine_args: EngineArgs,
        *,
        reset_kv_cache_on_model_state_dict_load: bool,
        stat_loggers: list | None = None,
    ):
        logger.info("Initializing LLMEngine from EngineArgs...")
        self._engine = LLMEngine.from_engine_args(
            engine_args, stat_loggers=stat_loggers
        )
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )

    @property
    def data_parallel_size(self) -> int:
        """The data-parallel size vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_size

    @property
    def data_parallel_rank(self) -> int:
        """The data-parallel rank vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_rank

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        ``load_prefetched_model_state_dict``. Safe to fill from any thread while no load runs."""
        return self._get_model()._prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        vLLM rejected."""
        rejected_requests: list[tuple[str, Exception]] = []
        for request in requests:
            prompt = {"prompt_token_ids": request.prompt_token_ids}
            if not self._reset_kv_cache_on_model_state_dict_load:
                # Salt by the pinned version so a request only reuses KV
                # computed under that version.
                prompt["cache_salt"] = str(request.min_policy_version)
            # vLLM validates a request (prompt length, sampling params, token ids) as it
            # admits it: render_cmpl and InputProcessor.process_inputs raise
            # VLLMClientError (e.g. VLLMValidationError, not a ValueError subclass) or
            # ValueError before LLMEngine.add_request registers the request with the
            # output processor and engine core, so a rejected request leaves no engine
            # state (recheck on a vLLM upgrade). The checks are deterministic, so every TP
            # rank of this replica rejects the same requests.
            try:
                # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized
                # prompts); the high-level entry stays resilient to vLLM internals vs
                # vllm.inputs.tokens_input.
                [engine_input] = self._engine.renderer.render_cmpl([prompt])
                self._engine.add_request(
                    request_id=request.request_id,
                    prompt=engine_input,
                    params=_build_sampling_params(request.sampling),
                )
            except (ValueError, VLLMClientError) as exc:
                rejected_requests.append((request.request_id, exc))
        return rejected_requests

    def has_unfinished_requests(self) -> bool:
        """Whether any admitted request is still running."""
        return self._engine.has_unfinished_requests()

    def step(self) -> list[FinishedRequest]:
        """Run one engine step, a TP collective, and return the requests it finished."""
        with torch.no_grad():
            request_outputs = self._engine.step()
        return [_finished_request(request_output) for request_output in request_outputs]

    def load_prefetched_model_state_dict(self) -> None:
        """Copy the prefetched model state dict from pinned CPU memory into the model, and
        reset the KV cache when configured."""
        model = self._get_model()
        model.load_state_dict(model._prefetched_model_state_dict)
        if self._reset_kv_cache_on_model_state_dict_load:
            # Always reset running requests too: the only reason to reset is a strict
            # recompute under the new weights. Keeping running requests' KV while hiding
            # old KV from new requests is already what the default (no reset) does via the salt.
            self._engine.reset_prefix_cache(
                reset_running_requests=True,
            )

    def shutdown(self) -> None:
        """Close renderer-local resources.

        With `external_launcher`, vLLM reuses the process group and actor lifetime that
        Monarch owns. Calling vLLM's internal `engine_core.shutdown()` can block while
        Monarch is also trying to stop the same actor mesh, so process teardown is left to
        `ProcMesh.stop()`.
        """
        renderer = getattr(self._engine, "renderer", None)
        if renderer is not None:
            logger.info("Shutting down vLLM renderer")
            renderer.shutdown()

    def _get_model(self):
        """The ``VLLMModelWrapper`` the engine runs."""
        return self._engine.model_executor.driver_worker.get_model()


def _build_sampling_params(sampling: SamplingConfig) -> SamplingParams:
    """Translate a `SamplingConfig` into vLLM `SamplingParams` (n=1).

    ``seed`` and ``stop_token_ids`` are carried on the ``SamplingConfig``
    (the controller fills ``stop_token_ids`` and the rollouter offsets
    ``seed`` per sample), so each sample in a group is a distinct ``n=1``
    request that stays diverse and bitwise-reproducible.

    The engine loads no tokenizer, so its ``eos_token_id`` is not a stop,
    but vLLM still adds the generation config's ``eos_token_id`` (checkpoint
    ``generation_config.json`` or the ``vllm_registry`` HF config).
    ``ignore_eos`` turns that off, so the renderer's ``stop_token_ids``
    (which include EOS) are the only stops.
    """
    return SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        n=1,  # always expects a single sample per request. Caller can call N times.
        stop_token_ids=sampling.stop_token_ids,
        # Drops the generation config's eos ids, which vLLM merges into
        # stop_token_ids even with skip_tokenizer_init.
        ignore_eos=True,
        seed=sampling.seed,
        logprobs=0,  # return only the sampled token's logprob (for the GRPO ratio)
        # Token ids in, token ids and logprob floats out: stops are token ids and nothing reads
        # text, so skip vLLM's per-token detokenization and per-token logprob dicts.
        detokenize=False,
        flat_logprobs=True,
        # Return each request's result once, when it is fully done, instead of streaming partial
        # outputs as tokens arrive.
        # TODO(async-rl): use RequestOutputKind.CUMULATIVE for exact per-token
        #   (start_token, version) boundaries; today we keep only the per-turn min/max.
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def _finished_request(request_output: RequestOutput) -> FinishedRequest:
    """Convert a finished ``RequestOutput`` into a ``FinishedRequest``."""
    # We enforce n=1 in sampling params -> exactly one CompletionOutput per finished request
    # Here we just sanity check it (a single engine.step may still finish several requests).
    if len(request_output.outputs) != 1:
        raise ValueError(
            f"expected n=1 (one sample per request), got "
            f"{len(request_output.outputs)} for {request_output.request_id}"
        )

    # flat_logprobs=True: vLLM returns logprobs as plain lists instead of one dict per token.
    # logprobs=0 keeps only the sampled token, so `.logprobs` has exactly one float per generated token.
    completion_output = request_output.outputs[0]
    return FinishedRequest(
        request_id=request_output.request_id,
        token_ids=list(completion_output.token_ids),
        token_logprobs=list(completion_output.logprobs.logprobs),
        finish_reason=completion_output.finish_reason,
        metrics_inputs=_extract_request_metrics_inputs(request_output),
    )


def _extract_request_metrics_inputs(
    request_output: RequestOutput,
) -> RequestMetricsInputs:
    """Pull the raw metric inputs off a finished ``RequestOutput``.

    For an `add_request` call, vLLM returns a RequestOutput carrying a single
    `RequestStateStats`.
    """
    stats = request_output.metrics
    if stats is None:
        return RequestMetricsInputs(
            num_cached_tokens=request_output.num_cached_tokens, has_stats=False
        )
    return RequestMetricsInputs(
        num_cached_tokens=request_output.num_cached_tokens,
        has_stats=True,
        queued_ts=stats.queued_ts,
        scheduled_ts=stats.scheduled_ts,
        first_token_ts=stats.first_token_ts,
        last_token_ts=stats.last_token_ts,
        first_token_latency=stats.first_token_latency,
        num_generation_tokens=stats.num_generation_tokens,
        num_preemptions=stats.num_preemptions,
    )
