# NVIDIA VSS blueprint — fit analysis for this machine

Notes from reading the blueprint's docs and source, and why a focused local
pipeline was installed instead of the enterprise stack.

- Blueprint card: <https://build.nvidia.com/nvidia/video-search-and-summarization/blueprintcard>
- Repository: <https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization>
- Cloned locally at `../vss` (depth 1, 3.4 GB, tag `3.3.0-rc0`)

## What the blueprint actually is

A reference platform for video analytics agents: live RTSP/ONVIF ingestion,
object detection and tracking, VLM-verified alerts, natural-language archive
search, and long-video summarisation. It is organised as ~30 microservices
across `services/` (agent, UI, real-time video intelligence, VIOS, alerts,
video-summarization, SDRC, analytics) with Docker Compose and Helm deployments
under `deploy/`.

The part relevant to "describe some videos" is **LVS** (`services/video-summarization`):
split a recording into chunks → caption each chunk with a VLM (RT-VLM) →
aggregate into a summary with an LLM. Its REST API is `POST /v1/summarize` on
port 38111. `video-describer/` implements that same pattern locally.

The two shipped models are **Cosmos3 Nano Reasoner** (VLM) and
**Nemotron 3.5 Lightning 30B-A3B** (LLM).

## Hardware reality check

| | Blueprint's requirement | This machine |
|---|---|---|
| Validated GPUs | H100, GB300, RTX PRO 6000 Blackwell (96 GB), L40S (48 GB), DGX Spark, IGX Thor, AGX Thor, RTX PRO 4500 BW | **RTX 5090 32 GB — not validated** |
| Unlisted GPUs | `HARDWARE_PROFILE=OTHER`, documented as *experimental* | what we'd have to use |
| Cosmos3 Nano Reasoner | >56 GB BF16; ~48 GB even in FP8 | **does not fit 32 GB** |
| Nemotron 3.5 Lightning 30B | ~78 GB BF16, ~45 GB INT4; NIM asks ≥66 GB/GPU | **does not fit 32 GB** |
| RAM | 128 GB host guidance (18 cores) | 59 GB |
| Disk | ~1 TB SSD guidance | 285 GB free |
| Docker | Engine 28.3.3 ≤ v < 29.5.0 | 29.5.2 — *but public nvcr.io pulls work (verified)* |

So: the default models cannot run here at all, and the full stack is heavier
than this host is sized for. Every local path requires swapping in a 7–12B model
regardless.

## Three real doors into the blueprint

1. **Full `dev-profile-lvs` stack** — ~18 containers: Elasticsearch, Kibana,
   Kafka, Logstash, Phoenix, VIOS (sensor/streamprocessing), SDRC, HAProxy, the
   agent UI, `lvs-server`, `rtvi-vlm`. Faithful, but RAM-tight here and
   experimental on this GPU.

2. **Standalone LVS** (`services/video-summarization/docker/deploy/compose.yaml`) —
   only `lvs` + `rtvi-vlm` + Elasticsearch (+ optional Kafka/Logstash). The
   `lvs` container itself needs **0 GPU**, 8 cores and ~32 GB RAM; GPU needs come
   entirely from its VLM/LLM peers. This is the sane way to run real LVS here.

3. **Everything local via vLLM** — RT-VLM supports
   `RTVI_VLM_MODEL_TO_USE=openai-compat` against `cosmos-reason1-7b`,
   `cosmos-reason2-8b` or `qwen3-vl-8b-instruct`, plus a documented
   bring-your-own-model path (`VLM_MODEL_TO_USE=vllm-compatible` + `MODEL_PATH`,
   see `skills/deployment/rtvi-byom-porting/references/vllm-porting.md`). Both
   the VLM and the summarisation LLM endpoints are configurable, so they can
   point at vLLM on this GPU. That removes the NGC-gated NIM images entirely —
   the first-party images are on public `ghcr.io/nvidia-ai-blueprints/vss`.

## Why the installed pipeline instead of door 2 or 3

For the stated goal — analyse and describe a handful of videos — installing
~50 GB of images plus Elasticsearch to run `POST /v1/summarize` buys little over
running the same chunk→caption→aggregate loop directly against a local model.
The installed pipeline is the same algorithm, ~10 GB of weights, no NGC key, no
gated images, and it starts in about three minutes.

## If you want the real LVS service later

Door 2, on top of the model server already running here:

1. Point RT-VLM at this vLLM server with
   `RTVI_VLM_MODEL_TO_USE=openai-compat` and `VIA_VLM_ENDPOINT=http://<host>:8000/v1`.
2. Point LVS's summariser at an OpenAI-compatible LLM with `LVS_LLM_HOST`,
   `LVS_LLM_PORT`, `LVS_LLM_MODEL_NAME` (a second vLLM, or the same endpoint).
3. Start the standalone compose with the `rtvi` profile, then
   `curl -X POST http://localhost:38111/v1/summarize` with `model`, `scenario`,
   `events`, and a video `url`.
4. Health check is `GET /v1/ready`; the model id must match `GET /models` exactly
   or you get a 400.

It needs free ports 38111, 8018, 9200 (plus 38112 for MCP), Elasticsearch
tuning for a 59 GB host, and about 50 GB of disk for images.

## Sources

- Prerequisites / GPU requirements: <https://docs.nvidia.com/vss/getting-started/prerequisites#development-profile-gpu-requirements>
- Configure the VLM (local vs remote, OpenAI-compatible): <https://docs.nvidia.com/vss/vision-agent/vss-agents/agent-customization/customize-models/configure-the-vlm>
- Configure the LLM: <https://docs.nvidia.com/vss/vision-agent/vss-agents/agent-customization/customize-models/configure-the-llm>
- Video summarization service: <https://docs.nvidia.com/vss/system-components/vision-microservices/video-summarization>
- Real-time VLM service: <https://docs.nvidia.com/vss/system-components/vision-microservices/real-time-vlm>
- NIM FAQ (self-host for dev vs. AI Enterprise for production): <https://docs.api.nvidia.com/nim/docs/product>
- RT-VLM byOM / vLLM porting: `vss/skills/deployment/rtvi-byom-porting/references/vllm-porting.md`
- LVS deployment reference: `vss/skills/operations/vss-summarize-video/references/deploy-lvs-service.md`
- Model choice: Qwen3-VL (<https://github.com/QwenLM/Qwen3-VL>), Video-MME
  (<https://video-mme.github.io/home_page.html>)
