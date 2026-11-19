"""把非 gated 的 CometKiwi 镜像（safetensors 格式）转成 comet 能加载的 Lightning ckpt。

背景：官方 Unbabel/wmt22-cometkiwi-da 是 gated 仓库，本机无 HF token → 401。
镜像 Supervache/wmt22-cometkiwi-da-safeformat 是同权重的 safetensors 版本
（hparams: class_identifier=unified_metric, pretrained_model=microsoft/infoxlm-large,
 license cc-by-nc-sa-4.0，与官方一致），但 comet 2.2.7 的 load_from_checkpoint
只认 checkpoints/model.ckpt，直接加载报 "Invalid checkpoint path"。

本脚本做一次格式转换并用 3 句验证。拿到官方 token 后应重跑一次一致性检查。

用法（comet venv）：
    & "D:\\HKStudy\\.venv-comet\\Scripts\\python.exe" src\\eval\\prepare_qe.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))

import torch  # noqa: E402
import yaml  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HF_HUB = Path(os.environ["HF_HOME"]) / "hub"
MIRROR = "models--Supervache--wmt22-cometkiwi-da-safeformat"
OUT_DIR = ROOT / "runs" / "qe_local" / "wmt22-cometkiwi-da-mirror"

PROBES = [
    {"src": "The medicine should be taken twice a day after meals.",
     "mt": "Le médicament doit être pris deux fois par jour après les repas."},
    {"src": "The medicine should be taken twice a day after meals.",
     "mt": "Le médicament doit être pris deux fois par jour après le voiture."},
    {"src": "I'm so done with this weather tbh",
     "mt": "J'en ai vraiment marre de ce temps"},
]


def find_snapshot() -> Path:
    snaps = sorted((HF_HUB / MIRROR / "snapshots").iterdir())
    if not snaps:
        raise SystemExit(f"未找到镜像快照：{HF_HUB / MIRROR}")
    return snaps[-1]


def main() -> None:
    snap = find_snapshot()
    st = snap / "checkpoints" / "model.safetensors"
    hp = snap / "hparams.yaml"
    print(f"[src] {st}  ({st.stat().st_size/1e9:.2f}GB)", flush=True)

    state = load_file(str(st))
    hparams = yaml.safe_load(hp.read_text(encoding="utf-8"))
    print(f"[hparams] class_identifier={hparams.get('class_identifier')} "
          f"pretrained={hparams.get('pretrained_model')} segments={hparams.get('input_segments')}",
          flush=True)
    print(f"[state] {len(state)} tensors, 首键 {list(state)[:3]}", flush=True)

    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "checkpoints" / "model.ckpt"
    torch.save({"state_dict": state, "hyper_parameters": hparams,
                "epoch": 2, "global_step": 164919,
                "pytorch-lightning_version": "1.8.2"}, str(ckpt_path))
    (OUT_DIR / "hparams.yaml").write_text(hp.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[out] {ckpt_path}  ({ckpt_path.stat().st_size/1e9:.2f}GB)", flush=True)

    del state
    from comet import load_from_checkpoint
    model = load_from_checkpoint(str(ckpt_path))
    print(f"[load] ok -> {type(model).__name__}", flush=True)

    try:
        out = model.predict(PROBES, batch_size=3, gpus=1, progress_bar=False)
    except TypeError:
        out = model.predict(PROBES, batch_size=3, accelerator="gpu", devices=1, progress_bar=False)
    scores = out[0] if isinstance(out, tuple) else (getattr(out, "scores", None) or out["scores"])
    print("[verify] 三句打分（好译/坏译/俚语）：", [round(float(s), 4) for s in scores], flush=True)
    assert float(scores[0]) > float(scores[1]), "好译分数应高于坏译，转换可能有误"
    print("[verify] 通过：好译 > 坏译", flush=True)
    print(f"\n在 score_comet.py 里用 --qe-model {ckpt_path} 调用", flush=True)


if __name__ == "__main__":
    main()
