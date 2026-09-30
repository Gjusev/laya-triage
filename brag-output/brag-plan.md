# Brag Plan: laya-triage

## What is this app?
Multilingual support-ticket triage on a local System 1 decision model: one ticket in, department,
intent, urgency, frustration, churn risk and refund flag out in a single forward pass, with a
measured confidence threshold deciding which tickets a human still reviews.

## The angle
Every triage demo answers "which department?". None answer the question that actually gates
automation: "when should the model back off?". This project publishes the answer as a measured
coverage/accuracy curve and picks its operating point from data: threshold 0.84, keep 60% of
tickets at 75% accuracy, escalate the rest with the reason attached. The video is a quiet
engineering flex: receipts, not promises.

## Hook (first 2-3 seconds)
A Spanish customer ticket types itself out, character by character:
"me cobraron dos veces, reembolsen o cancelo"
Under it, small mono type fades in: `one forward pass · any language · no API calls`
The typing IS the product: any language goes in.

## Key moments (the middle)
- The hierarchy reveal: the flat 77-intent list collapses/fades, replaced by 12 cluster chips;
  one chip highlights (`card payment problems`), then the exact intent lands
  (`transaction_charged_twice`) — structure alone is worth +14.5 points (36.5% → 51.0%).
- The four signal rows arriving one by one in the same pass: urgency 1.8 · frustration 1.9 ·
  churn 0.87 · refund 0.91 — one card sound each.
- The escalation curve drawing itself: coverage vs accuracy, the 0.84 threshold ticking in, and
  the verdict flipping to `escalate to human — reason attached`.

## Outro / punchline
"At what confidence do you let a model close a ticket?
0.84. Measured."
Then the wordmark `laya-triage` + `Apache 2.0 · github.com/Gjusev/laya-triage`. Soft final hit,
bed fades.

## User flow worth showing
Paste a raw ticket (Spanish) → watch the triage result assemble (department badge, intent,
four signals) → watch the confidence gate decide auto-handle vs escalate. Entry → triage →
verdict, recreated as Scenes 1-2. The metrics view (the curve) is the product's own receipt
and closes the argument as Scene 3.

## Tone
- Preset: `polished`
- Creative direction: quiet engineering flex — a demo that publishes its own error bars
- Interpretation: longer holds, fewer scenes, restrained motion; numbers carry the drama;
  typography is editorial (serif display + mono data), palette is warm paper with pastel
  semantics. No SaaS gloss, no neon.

## Format: landscape — 1920x1080
## Duration: 20 seconds

## Visual identity (from the project)
- Background: `#F7F6F3` (warm paper)
- Ink / text: `#111111`
- Accents (semantic pastels only): auto `#EDF3EC`/`#346538` · escalate `#FDEBEC`/`#9F2F2D` ·
  data `#E1F3FE`/`#1F6C9F` · note `#FBF3DB`/`#956400`; hairlines `#EAEAEA`
- Display font: Newsreader (editorial serif; italic for emphasis)
- Data font: JetBrains Mono
- Body font: Helvetica Neue / Switzer stack
- Strongest visual element: the escalation curve with the 0.84 threshold tick, and pale pastel
  pill tags on white cards with 1px hairline borders

## Share copy (draft)
A ticket triage demo that answers the question demos skip: at what confidence does the model
get to act? 0.84 — measured, curve included.

## Audio direction
- Role: warm minimal bed, sparse professional accents
- Music: bundled warm/organic electronic bed if present; low posture, fade-in under the typing,
  gentle swell under the curve, fade out on the outro
- Music treatment: start ~0.3s in, quiet under Scene 1 (typing is the lead), rise slightly for
  Scene 3 (the receipts), tail fade to silence by the wordmark
- Music cue guidance: preset cues to be read from the bundled track at composition time; target
  one strong cue near the curve draw (Scene 3) and one at the outro wordmark; sequential signal
  rows in Scene 2 reveal roughly every other beat, never faster than 0.5s apart
- Audio-reactive treatment: none — restraint is the aesthetic
- SFX posture: sparse, motion-matched; keyboard ticks for the typed ticket, soft card sound per
  signal row, one distinct tick when the 0.84 threshold lands, one soft final hit on the
  wordmark
- Audio-coupled moments: typed hook text; card-by-card signal rows; threshold tick; count-up of
  "+14.5"
- Restraint rule: no whooshes, no risers, no bass drops; audio never louder than the reading

## Storyboard

### Scene 1 — the ticket types itself — 3.5s
Warm paper background. The Spanish ticket types character by character in a white card with a
1px hairline border (like the app's input). As it finishes, small mono caption fades in:
`one forward pass · any language · no API calls`.
Sequential/interaction: yes — simulated typing of the hook line.
Audio intent: intimacy, focus; the bed is nearly silent under the keys.
Audio-coupled idea: subtle key ticks on typing; one soft tick when the caption lands.
Transition mood: soft → Scene 2

### Scene 2 — the triage assembles — 6s
Same card scales slightly; the result assembles around it: department badge `payments_cash`
(pale blue pill), cluster chip `card payment problems`, then intent `transaction_charged_twice`
in mono; below, the four signal rows arrive one by one (urgency 1.8 / frustration 1.9 /
churn 0.87 / refund 0.91) with tiny bars. A hairline divider, then the confidence pair:
cluster 0.82 · intent 0.71.
Sequential/interaction: yes — five elements arrive one by one (badges/rows), each with a soft
card sound, holding to read (~0.8s each, staged so all remain on screen).
Audio intent: the product feeling alive and precise.
Audio-coupled idea: card-by-card arrival sounds roughly every other beat.
Transition mood: clean → Scene 3

### Scene 3 — the receipts — 6s
Two moments. First, the comparison: `36.5%` (flat 77-way, muted) vs `51.0%` (hierarchical,
ink) with `+14.5 points from structure alone` counting up in serif. Then the escalation curve
draws itself as an SVG line (coverage → accuracy), the 0.84 threshold ticks in as a vertical
hairline, and the operating point dots: `keep 60% · 75% accurate · escalate the rest, reason
attached` with a pale red `escalate` pill and a pale green `auto-handle` pill.
Sequential/interaction: yes — count-up, then line-draw, then threshold tick (three staged beats).
Audio intent: the receipt landing; measured confidence.
Audio-coupled idea: one distinct tick on the 0.84 threshold; gentle swell under the line draw.
Transition mood: soft → Scene 4

### Scene 4 — the answer — 4.5s
Editorial serif, tight tracking, near-black on paper:
"At what confidence do you let a model close a ticket?"
Then italic serif: `0.84. Measured.`
Wordmark `laya-triage` in mono, small, with `Apache 2.0 · github.com/Gjusev/laya-triage`.
Sequential/interaction: yes — question holds, answer lands, wordmark fades in.
Audio intent: the punchline; quiet final hit, bed fades to silence.
Audio-coupled idea: one soft final hit on the answer line.
Transition mood: soft hold to end

**Music mood for this video:** warm, restrained, minimal electronic
**Audio summary:** near-silent keys under the typed hook, a low warm bed rising gently through
the triage assembly, one swell under the curve, and a quiet final hit as the number lands —
audio as lab-notebook calm, never as hype.
