"""One-shot generator for finetune/laya_triage_banking77.ipynb.

Run: python finetune/_build_notebook.py
Regenerates the notebook from scratch so the .ipynb JSON never has to be
hand-edited. Kept in the repo for the same reason as any generated artifact's
source: reproducibility.
"""

import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


TRAINER = r'''%%writefile /kaggle/working/train_ddp.py
import os, sys, time, json, random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward

def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
            "label": torch.tensor([it["label"] for it in items])}

def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())

def main():
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir, output_dir = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    # keep the hub max_len/head_max_len (512/192): training sequences must match
    # what serving builds, or the fine-tune optimizes a different head layout

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load("/kaggle/working/train_items.pt", weights_only=False)
    # hold the calibration slice out of training (calibrate on unseen items)
    calib_items = all_items[::15][:400]
    calib_ids = {id(x) for x in calib_items}
    train_pool = [it for it in all_items if id(it) not in calib_ids]
    # equal count per rank so every DDP step syncs cleanly
    train_pool = train_pool[: len(train_pool) // world_size * world_size]
    my_items = train_pool[rank::world_size]

    EPOCHS = 3
    MICRO_BATCH = 8
    GRAD_ACCUM = 4
    GROUP_SIZE = 4
    LR_ENCODER, LR_HEAD = 2.5e-5, 1.0e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}
    ], weight_decay=0.01)
    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(f"DDP training: {len(all_items)} items | {len(my_items)} per rank | {EPOCHS} epochs")
    t0 = time.time()
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                        batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                        batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            scaler.scale(loss).backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if rank == 0 and n_batches % 200 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS} step {n_batches} loss {loss.item()*GRAD_ACCUM:.4f} reward {r.mean().item():.3f}")
        if rank == 0:
            print(f"=== epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.1f}s | avg loss {epoch_loss/max(1,n_batches):.4f} ===")
        dist.barrier()
        if rank == 0:  # rolling checkpoint survives Kaggle session resets
            ckpt = os.path.join(output_dir, "checkpoint_latest")
            os.makedirs(ckpt, exist_ok=True)
            save_file({kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()},
                      os.path.join(ckpt, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt, "encoder"))
            tok.save_pretrained(os.path.join(ckpt, "tokenizer"))
            json.dump({"epoch": epoch + 1, "avg_loss": epoch_loss / max(1, n_batches)},
                      open(os.path.join(ckpt, "checkpoint_meta.json"), "w"), indent=2)
    dist.barrier()
    if rank == 0:  # final save + per-type temperature calibration
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        preds = []
        with torch.no_grad():
            for c in range(0, len(calib_items), 16):
                cb = collate_train_batch(calib_items[c:c+16], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                     cb["marker_pos"].to(device), cb["marker_mask"].to(device), cb["qtype"].to(device))
                l_np = l_sub.float().cpu().numpy()
                preds += [(it["qtype"], l_np[ri, :len(it["markers"])], it["target"]) for ri, it in enumerate(calib_items[c:c+16])]
        temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_, z, t in preds if q_ == qt]
                if sel:
                    temps[qt] = fit_one_temp(sel)
            print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in temps])
        except Exception as e:
            print("temperature fitting fallback:", e)
        os.makedirs(output_dir, exist_ok=True)
        save_file({kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()},
                  os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-triage-banking77"
        cfg.pop("temperature_by_options", None)  # stale hub buckets would override the fitted list
        cfg["temperature"] = temps
        json.dump(cfg, open(os.path.join(output_dir, "rl_agent_config.json"), "w"), indent=2)
        print(f"model saved to {output_dir}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

EVAL_CELL = r'''import json, random, csv, io, collections, urllib.request
import laya
from laya_triage import schema
from laya_triage.escalation import coverage_accuracy_curve, choose_threshold

agent_ft = laya.Agent(OUTPUT_DIR, device="cuda")

# test split + the same deterministic sample as the repo eval (seed 13, limit 200);
# inlined because the pip-installed wheel ships only the laya_triage package
test_csv = urllib.request.urlopen(
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
).read().decode("utf-8")
all_test = [{"text": r["text"], "label": r["category"]} for r in csv.DictReader(io.StringIO(test_csv))]
rows = random.Random(13).sample(all_test, 200) if len(all_test) > 200 else list(all_test)
texts = [r["text"] for r in rows]

# hierarchical two-stage with the fine-tuned checkpoint
stage1 = agent_ft.predict_batch([{"message": t} for t in texts], schema.coarse_questions())
clusters = [o["answers"]["cluster"]["choice"] for o in stage1]
by_cluster = collections.defaultdict(list)
for i, c in enumerate(clusters):
    by_cluster[c].append(i)
intents = {}
for c, idxs in by_cluster.items():
    outs = agent_ft.predict_batch([{"message": texts[i]} for i in idxs], schema.fine_questions(c))
    for i, o in zip(idxs, outs):
        intents[i] = (o["answers"]["intent"]["choice"], o["answers"]["intent"]["answer_confidence"])

records = []
for i, row in enumerate(rows):
    intent, intent_conf = intents[i]
    records.append({"label": row["label"], "pred": intent, "cluster": clusters[i],
                    "cluster_conf": stage1[i]["answers"]["cluster"]["answer_confidence"],
                    "intent_conf": intent_conf, "correct": intent == row["label"]})

ft_acc = sum(r["correct"] for r in records) / len(records)
coarse_acc = sum(schema.INTENT_TO_CLUSTER[r["label"]] == r["cluster"] for r in records) / len(records)
curve = coverage_accuracy_curve([{"confidence": min(r["cluster_conf"], r["intent_conf"]), "correct": r["correct"]} for r in records])
chosen = choose_threshold(curve, target_accuracy=0.75)

# published zero-shot baselines from the repo's artifact (same sample, same seed)
import urllib.request
B77 = json.load(urllib.request.urlopen(
    "https://raw.githubusercontent.com/Gjusev/laya-triage/main/evals/results/banking77.json"))
print(f"zero-shot direct  : {B77['summary']['direct_accuracy']:.3f}")
print(f"zero-shot hier    : {B77['summary']['hierarchical_accuracy']:.3f}")
print(f"fine-tuned hier   : {ft_acc:.3f} (coarse {coarse_acc:.3f})")
if chosen:
    print(f"escalation@75%    : threshold {chosen.threshold:.2f}, coverage {chosen.coverage:.1%}")
else:
    print("escalation@75%    : target not reachable on this sample")

pairs = collections.Counter((r["label"], r["pred"]) for r in records if not r["correct"])
top_failures = [(a, b, n) for (a, b), n in pairs.most_common(10)]
print("top confusions (gold -> pred xN):", top_failures)
'''

SIGNALS_CELL = r'''import json, urllib.request
hl = [json.loads(l) for l in urllib.request.urlopen(
    "https://raw.githubusercontent.com/Gjusev/laya-triage/main/data/hand_labeled_200.jsonl"
).read().decode("utf-8").splitlines() if l.strip()]
en_rows = [r for r in hl if r["lang"] == "en"]  # the fine-tuned checkpoint is English-only
outs = agent_ft.predict_batch([{"message": r["text"]} for r in en_rows], schema.coarse_questions())
u_mae = sum(abs(o["answers"]["urgency"]["score"] - r["urgency"]) for o, r in zip(outs, en_rows)) / len(en_rows)
f_mae = sum(abs(o["answers"]["frustration"]["score"] - r["frustration"]) for o, r in zip(outs, en_rows)) / len(en_rows)
print(f"fine-tuned signals on en hand-labeled (n={len(en_rows)}): urgency MAE {u_mae:.3f}, frustration MAE {f_mae:.3f}")
published = json.load(urllib.request.urlopen(
    "https://raw.githubusercontent.com/Gjusev/laya-triage/main/evals/results/hand_labeled.json"))["signals"]
print("published zero-shot (all languages):", {k: published[k] for k in ("urgency_mae", "frustration_mae")})
'''

PUSH_CELL = r'''import os
from huggingface_hub import HfApi
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    token = None

NEW_REPO = "YOUR_USERNAME/laya-triage-banking77"  # TODO: set before running

failures_md = "\n".join(f"- `{a}` predicted as `{b}` (x{n})" for a, b, n in top_failures)
readme = f"""---
license: apache-2.0
library_name: transformers
tags:
- laya
- system-one
- ticket-triage
- banking77
- hierarchical-classification
metrics:
- accuracy
---
# laya-triage-banking77

Laya (421M) fine-tuned on the BANKING77 training split for hierarchical
ticket triage: a coarse choice over 12 clusters plus a fine choice over
the winning cluster's intents, trained jointly. Part of the
[laya-triage](https://github.com/Gjusev/laya-triage) project.

## Results (200-ticket seed-13 test sample)

| Configuration | Intent accuracy |
|---|---|
| zero-shot flat 77-way | {B77['summary']['direct_accuracy']:.3f} |
| zero-shot hierarchical | {B77['summary']['hierarchical_accuracy']:.3f} |
| fine-tuned hierarchical (this model) | {ft_acc:.3f} |

Coarse accuracy {coarse_acc:.3f}. Escalation threshold at the 75% accuracy
target: {escalation_md}.

## What was trained and what was not

- Trained: the two routing decisions (cluster choice, intent choice).
- NOT trained: the auxiliary signal heads (urgency, frustration, churn risk,
  refund requested) - BANKING77 carries no such labels. Signal MAE
  before/after is checked in the training notebook and published in the repo.
- English only: the base checkpoint is the English laya model; the repo
  documents in-domain coarse accuracy degradation for other languages, which
  this fine-tune does not address.

## Where it fails (top confusions on the eval sample)

{failures_md}

## Usage

```python
import laya
agent = laya.load("YOUR_USERNAME/laya-triage-banking77")
```
See the laya-triage repo for the full pipeline (escalation policy,
multilingual routing, signals).

## License
Apache 2.0. Base model by Convai Innovations; fine-tune by the laya-triage
project.
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme)

if not token:
    print("HF_TOKEN secret not set: skipping the Hub upload.")
    print(f"Weights and the generated model card stay in {OUTPUT_DIR} (kernel output);")
    print("publish them later by re-running this cell with the secret attached.")
else:
    api = HfApi(token=token)
    api.create_repo(NEW_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=NEW_REPO, repo_type="model",
                      commit_message=f"laya fine-tuned on BANKING77 routing: intent accuracy {ft_acc:.3f}")
    print(f"published: https://huggingface.co/{NEW_REPO}")
'''


def build():
    cells = [
        md(
            "# Fine-tuning laya on BANKING77 for hierarchical ticket triage (Kaggle T4 x2, DDP)",
            "",
            "Adapted from laya's official notebook",
            "([laya_finetune_typed_decisions_2xT4_kaggle.ipynb]",
            "(https://github.com/NandhaKishorM/laya/blob/master/notebooks/"
            "laya_finetune_typed_decisions_2xT4_kaggle.ipynb)) for the laya-triage project.",
            "It fine-tunes the English laya checkpoint (`convaiinnovations/laya`, 421M)",
            "on the BANKING77 training split so the two routing decisions of the",
            "hierarchy - the coarse cluster choice and the fine intent choice - are",
            "trained together, then evaluates against the zero-shot baselines",
            "published in the repo and pushes an honest model card to HuggingFace.",
            "",
            "**Scope decisions (documented in the repo README):**",
            "- Only the routing decisions (cluster, intent) are fine-tuned. BANKING77",
            "  has no labels for urgency/frustration/churn/refund, so the auxiliary",
            "  signals keep their base weights; the notebook re-checks signal MAE on",
            "  the project's hand-labeled set afterwards (the plan's stated risk).",
            "- The multilingual checkpoint is NOT fine-tuned (no non-English BANKING77",
            "  training data exists); in-domain per-language degradation is documented",
            "  in the repo and left as follow-up work.",
            "",
            "### Kaggle notebook settings",
            "- **Accelerator:** GPU T4 x2 (both GPUs required for DDP)",
            "- **Internet:** On",
            "- **Secrets:** add your HuggingFace write token as `HF_TOKEN`",
            "- Output goes to `/kaggle/working/laya_triage_banking77`",
        ),
        code([
            "!nvidia-smi\n",
            "import os, torch\n",
            "\n",
            "n_gpu = torch.cuda.device_count()\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()} | Visible GPUs: {n_gpu}\")\n",
            "assert n_gpu >= 2, \"Set Accelerator to GPU T4 x2 in the notebook options\"\n",
            "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
        ]),
        md(
            "## 1. Install dependencies",
            "The laya-triage package provides the exact question builders the pipeline",
            "uses, so training and evaluation share one definition.",
            "",
            "**Prerequisite (TODO(push)):** the pip install below and the artifact",
            "fetches in cells 5-6 resolve against `origin/main` on GitHub. Push the",
            "repo (all phases committed locally) before running this notebook on",
            "Kaggle, or the install delivers a package without `laya_triage.schema",
            "` and the artifact URLs 404.",
        ),
        code([
            "!pip install -q -U \"laya>=0.3\" \"transformers>=4.48.0\" datasets safetensors huggingface_hub accelerate pandas\n",
            "!pip install -q git+https://github.com/Gjusev/laya-triage.git\n",
            "import laya, laya_triage\n",
            "print(\"laya\", laya.__version__, \"| laya_triage\", laya_triage.__version__)\n",
        ]),
        md(
            "## 2. Build training items from BANKING77",
            "Each ticket contributes two sequences: the coarse cluster choice (gold =",
            "the intent's cluster) and the fine intent choice restricted to that",
            "cluster (gold = the intent). Targets are one-hot, the format the official",
            "trainer consumes.",
        ),
        code([
            "import csv, io, json, os, urllib.request, torch\n",
            "from transformers import AutoTokenizer\n",
            "from huggingface_hub import snapshot_download\n",
            "from laya.agent import _fix_tokenizer_config\n",
            "from laya.common import build_sequence, render_options, QTYPES\n",
            "from laya_triage import schema\n",
            "\n",
            "MODEL_ID = \"convaiinnovations/laya\"\n",
            "model_dir = snapshot_download(MODEL_ID)\n",
            "_fix_tokenizer_config(model_dir)\n",
            "tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, \"tokenizer\"))\n",
            "cfg = json.load(open(os.path.join(model_dir, \"rl_agent_config.json\")))\n",
            "\n",
            "BASE = \"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data\"\n",
            "rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(BASE + \"/train.csv\").read().decode(\"utf-8\"))))\n",
            "print(f\"BANKING77 train: {len(rows)} tickets\")\n",
            "\n",
            "def build_item(state, q, gold_label):\n",
            "    t = q[\"type\"]\n",
            "    crit = q.get(\"criteria\", {})\n",
            "    keys = list(crit.keys())\n",
            "    target = [1.0 if k == gold_label else 0.0 for k in keys]\n",
            "    label = target.index(max(target))\n",
            "    k_options = len(render_options({\"t\": t, \"crit\": crit}))\n",
            "    seq, markers = build_sequence(\n",
            "        tok, state, {\"t\": t, \"ins\": q[\"instructions\"], \"crit\": crit},\n",
            "        cfg[\"max_len\"], cfg[\"head_max_len\"],\n",
            "    )\n",
            "    if len(markers) != k_options:\n",
            "        return None\n",
            "    return {\"ids\": seq, \"markers\": markers, \"qtype\": QTYPES[t],\n",
            "            \"target\": target, \"label\": label}\n",
            "\n",
            "items = []\n",
            "for row in rows:\n",
            "    state = {\"message\": row[\"text\"]}\n",
            "    intent = row[\"category\"]\n",
            "    cluster = schema.INTENT_TO_CLUSTER[intent]\n",
            "    coarse = build_item(state, schema.coarse_questions()[\"cluster\"], cluster)\n",
            "    fine = build_item(state, schema.fine_questions(cluster)[\"intent\"], intent)\n",
            "    items += [it for it in (coarse, fine) if it]\n",
            "\n",
            "print(f\"Preprocessed {len(items)} training sequences "
            "({len(items) // 2} tickets x 2 routing decisions).\")\n",
            "torch.save(items, \"/kaggle/working/train_items.pt\")\n",
        ]),
        md(
            "## 3. DDP training script",
            "Trainer from the official notebook (RLCD policy gradients with proper",
            "scoring rules + soft cross-entropy guidance, rolling checkpoints,",
            "post-training temperature calibration); only the output metadata changes.",
        ),
        code(TRAINER.splitlines(keepends=True)),
        md(
            "## 4. Launch training",
            "~20,000 sequences over 3 epochs on 2 T4s; the rolling checkpoint after",
            "each epoch survives Kaggle session resets.",
        ),
        code([
            "OUTPUT_DIR = \"/kaggle/working/laya_triage_banking77\"\n",
            "!torchrun --standalone --nproc_per_node=2 /kaggle/working/train_ddp.py {model_dir} {OUTPUT_DIR}\n",
        ]),
        md(
            "## 5. Evaluate: fine-tuned vs the published zero-shot baselines",
            "Same seed-13 test sample and same question builders as the repo eval,",
            "so the three configurations are directly comparable. Also computes the",
            "top confusion pairs for the model card's where-it-fails section.",
        ),
        code(EVAL_CELL.splitlines(keepends=True)),
        md(
            "### Signal regression check (the plan's stated risk)",
            "Routing-only training should leave the signal heads close to their",
            "zero-shot behavior; this verifies it on the English subset of the",
            "hand-labeled set.",
        ),
        code(SIGNALS_CELL.splitlines(keepends=True)),
        md(
            "## 6. Publish to HuggingFace with an honest model card",
            "The card is generated from THIS run's numbers (never hardcoded), states",
            "what was trained and what was not, and names the top failure modes.",
            "The card is always written to the kernel output; the Hub upload runs",
            "only when the HF_TOKEN secret is attached, otherwise it is skipped",
            "without failing the run. Set `NEW_REPO` to your username to publish.",
        ),
        code(PUSH_CELL.splitlines(keepends=True)),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = Path(__file__).parent / "laya_triage_banking77.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {out} ({n_code} code cells)")


if __name__ == "__main__":
    build()
