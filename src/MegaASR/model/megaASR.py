from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file

from .Qwen3_ASR import Qwen3ASR
from .router import AudioQualityRouter


class LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = True) -> None:
        self.keep_delta_on_gpu = keep_delta_on_gpu
        self.items: list[dict[str, Any]] = []
        self.active = False

    def _load_adapter_state(self, adapter_dir: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
        adapter_dir = str(adapter_dir)
        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            return safe_load_file(safetensors_path)
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")

        raise FileNotFoundError(
            "Cannot find adapter_model.safetensors or adapter_model.bin under "
            f"{adapter_dir}"
        )

    def _load_adapter_config(self, adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
        config_path = os.path.join(str(adapter_dir), "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Cannot find adapter_config.json under {adapter_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        for prefix in ("base_model.model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        return name

    def _split_lora_key(self, key: str) -> tuple[str | None, str | None]:
        key = self._normalize_module_name(key)

        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                module_name = key.split(marker)[0]
                kind = "A" if marker == ".lora_A." else "B"
                return module_name, kind

        return None, None

    def add_adapter(
        self,
        parent_module: torch.nn.Module,
        adapter_dir: str | os.PathLike[str],
        name: str,
        strip_prefixes: list[str] | None = None,
    ) -> None:
        config = self._load_adapter_config(adapter_dir)
        state = self._load_adapter_state(adapter_dir)

        lora_alpha = config.get("lora_alpha", 1)
        rank = config.get("r")
        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

        module_dict = dict(parent_module.named_modules())
        grouped: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in state.items():
            module_name, kind = self._split_lora_key(key)
            if module_name is None or kind is None:
                continue

            if strip_prefixes:
                for prefix in strip_prefixes:
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix) :]

            grouped.setdefault(module_name, {})[kind] = tensor.cpu()

        loaded = 0
        missing = []

        for module_name, pair in grouped.items():
            if "A" not in pair or "B" not in pair:
                continue
            if module_name not in module_dict:
                missing.append(module_name)
                continue

            module = module_dict[module_name]
            if not hasattr(module, "weight"):
                missing.append(module_name)
                continue

            a_matrix = pair["A"].float()
            b_matrix = pair["B"].float()
            adapter_rank = rank_pattern.get(module_name, rank)
            if adapter_rank is None:
                adapter_rank = a_matrix.shape[0]
            adapter_alpha = alpha_pattern.get(module_name, lora_alpha)
            scaling = float(adapter_alpha) / float(adapter_rank)

            delta = torch.matmul(b_matrix, a_matrix) * scaling
            weight = module.weight

            if fan_in_fan_out:
                delta = delta.T

            if delta.shape != weight.shape:
                try:
                    delta = delta.reshape(weight.shape)
                except Exception:
                    missing.append(
                        f"{module_name}: delta shape {tuple(delta.shape)} != "
                        f"weight shape {tuple(weight.shape)}"
                    )
                    continue

            delta = delta.to(dtype=weight.dtype)
            if self.keep_delta_on_gpu:
                delta = delta.to(device=weight.device)

            self.items.append(
                {
                    "name": name,
                    "module_name": module_name,
                    "weight": weight,
                    "delta": delta,
                }
            )
            loaded += 1

        if loaded == 0:
            raise ValueError(f"No LoRA delta loaded for adapter {name} from {adapter_dir}")

        if missing:
            warnings.warn(
                f"LoRA adapter {name} loaded {loaded} deltas, "
                f"missing {len(missing)} modules. Examples: {missing[:5]}",
                stacklevel=2,
            )

    @torch.no_grad()
    def set_active(self, active: bool) -> float:
        if self.active == active:
            return 0.0

        start = time.perf_counter()
        sign = 1.0 if active else -1.0

        for item in self.items:
            weight = item["weight"]
            delta = item["delta"]
            if delta.device != weight.device:
                delta = delta.to(device=weight.device)
            weight.data.add_(delta, alpha=sign)

        self.active = active
        return time.perf_counter() - start


class MegaASR:
    NAME = "Mega-ASR"
    DEFAULT_MODEL_DIR = Qwen3ASR.DEFAULT_MODEL_DIR
    DEFAULT_LORA_DIR = "ckpt/Mega-ASR/ckpt/A2S-SFT-lora/mega-asr-merged"
    DEFAULT_ROUTER_CHECKPOINT = AudioQualityRouter.DEFAULT_CHECKPOINT
    DOWNLOAD_URLS = {
        "lora": None,
        "router": None,
    }

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        lora_dir: str | os.PathLike[str] | None = None,
        router_checkpoint: str | os.PathLike[str] | None = None,
        routing_enabled: bool = True,
        fallback_use_lora: bool = True,
        quality_threshold: float = 0.5,
        device_map: str | None = None,
        quality_device: str | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        keep_delta_on_gpu: bool = True,
        **model_kwargs: Any,
    ) -> None:
        self.model_path = str(Path(model_path or self.DEFAULT_MODEL_DIR).expanduser())
        self.lora_dir = str(Path(lora_dir or self.DEFAULT_LORA_DIR).expanduser())
        self.router_checkpoint = str(
            Path(router_checkpoint or self.DEFAULT_ROUTER_CHECKPOINT).expanduser()
        )
        self.routing_enabled = routing_enabled
        self.fallback_use_lora = fallback_use_lora

        self.stats = {"total": 0, "use_base": 0, "use_lora": 0}
        self.switch_times: list[dict[str, float | str]] = []

        self.router = None
        if self.routing_enabled:
            try:
                self.router = AudioQualityRouter(
                    checkpoint_path=self.router_checkpoint,
                    device=quality_device,
                    threshold=quality_threshold,
                )
            except FileNotFoundError as exc:
                warnings.warn(
                    f"{exc} Routing is disabled and fallback_use_lora="
                    f"{self.fallback_use_lora}.",
                    stacklevel=2,
                )
                self.routing_enabled = False

        self.asr = Qwen3ASR(
            model_path=self.model_path,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

        if not hasattr(self.asr.model.model, "thinker"):
            raise ValueError("Qwen3-ASR inner model does not have attribute `thinker`.")

        self.lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=keep_delta_on_gpu)
        self._load_loras()
        self._set_lora(self.fallback_use_lora)

    @classmethod
    def download(cls, name: str, target_dir: str | os.PathLike[str]) -> str:
        url = cls.DOWNLOAD_URLS.get(name)
        if not url:
            raise NotImplementedError(f"Download URL for {name} is not set yet.")

        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=url,
            local_dir=str(Path(target_dir).expanduser()),
            local_dir_use_symlinks=False,
        )

    def _load_loras(self) -> None:
        self.lora_switch.add_adapter(
            parent_module=self.asr.model.model,
            adapter_dir=self.lora_dir,
            name="mega_asr_lora",
        )

    def _set_lora(self, active: bool) -> None:
        elapsed = self.lora_switch.set_active(active)
        if elapsed > 0:
            direction = "base_to_lora" if active else "lora_to_base"
            self.switch_times.append({"direction": direction, "time": elapsed})

    @staticmethod
    def _unwrap_audio(audio: Any) -> Any:
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            return audio[0]
        return audio

    def _route(self, audio: Any) -> tuple[bool, float | None, str]:
        if self.routing_enabled and self.router is not None:
            is_dirty, dirty_prob = self.router.predict(audio)
            return is_dirty, dirty_prob, "router"

        return self.fallback_use_lora, None, "fallback"

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        return_route: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        audio = self._unwrap_audio(audio)
        use_lora, dirty_prob, route_source = self._route(audio)

        self._set_lora(use_lora)
        result = self.asr.infer(
            audio,
            language=language,
            return_objects=return_objects,
            **transcribe_kwargs,
        )

        self.stats["total"] += 1
        if use_lora:
            self.stats["use_lora"] += 1
        else:
            self.stats["use_base"] += 1

        if return_route:
            return {
                "text": result,
                "use_lora": use_lora,
                "dirty_prob": dirty_prob,
                "route_source": route_source,
            }

        return result

    def infer_with_lora(self, audio: Any, **kwargs: Any) -> Any:
        self._set_lora(True)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    def infer_without_lora(self, audio: Any, **kwargs: Any) -> Any:
        self._set_lora(False)
        return self.asr.infer(self._unwrap_audio(audio), **kwargs)

    @torch.no_grad()
    def batch_infer(self, audios: list[Any], **kwargs: Any) -> list[Any]:
        audio_paths = [self._unwrap_audio(audio) for audio in audios]
        routes = [self._route(audio) for audio in audio_paths]

        base_indices = [idx for idx, route in enumerate(routes) if not route[0]]
        lora_indices = [idx for idx, route in enumerate(routes) if route[0]]

        results: list[Any] = [None] * len(audio_paths)
        groups = [("lora", lora_indices), ("base", base_indices)]
        if not self.lora_switch.active:
            groups = [("base", base_indices), ("lora", lora_indices)]

        for mode, indices in groups:
            if not indices:
                continue

            use_lora = mode == "lora"
            self._set_lora(use_lora)

            for idx in indices:
                results[idx] = self.asr.infer(audio_paths[idx], **kwargs)
                if use_lora:
                    self.stats["use_lora"] += 1
                else:
                    self.stats["use_base"] += 1

        self.stats["total"] += len(audio_paths)
        return results


def get_mega_asr(*args: Any, **kwargs: Any) -> MegaASR:
    return MegaASR(*args, **kwargs)


def get_Mega_ASR(*args: Any, **kwargs: Any) -> MegaASR:
    return get_mega_asr(*args, **kwargs)