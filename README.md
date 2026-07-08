# J-Space-Replay

Upload a short video, ask a question, then replay the model's decoded internal
concept readouts synced to the video timeline.

> **Demo-quality interpretability.** Lens readouts are noisy, single-token,
> and unvalidated on vision-language models. The J-lens method was validated
> on Claude text models only; this tool extrapolates it to a VLM. Not suitable
> for mechanistic claims.

## Docs

- [SPEC.md](SPEC.md) — product spec: what it is, trace format, UI, honest framing
- [PLAN.md](PLAN.md) — production plan: prerequisites, milestones M0–M5, risk register, J-lens borrow plan

## Target hardware

RTX 5070 Ti (16 GB) / Windows 11 · Qwen2.5-VL-7B-Instruct 4-bit (bitsandbytes NF4) ·
PyTorch cu128 + HF transformers (SDPA) · FastAPI backend · React frontend.

## References

- Anthropic, [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) — research anchor (J-lens / J-space)
- [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36) (Apache 2.0) — implementation pattern for the analytic J-lens fit
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Apache 2.0)

## Status

Pre-implementation. See [PLAN.md](PLAN.md) — currently at Milestone 0.
