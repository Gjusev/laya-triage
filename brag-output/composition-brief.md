# Hyperframes Composition Brief: laya-triage

## Objective
Create a short launch-style brag video for laya-triage.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `C:/Code Main/laya-triage`
- Primary files read: README.md, app.py, src/laya_triage/schema.py + pipeline.py,
  evals/results/banking77.json, docs/assets/how-it-works.html
- Product name: laya-triage
- Tagline / strongest claim: "At what confidence do you let a model close a ticket? 0.84.
  Measured."
- Key UI or visual moment to recreate: the triage result card assembling (department pill,
  intent, four signal rows with tiny bars, confidence pair) and the escalation curve with the
  0.84 threshold tick
- Copy that must appear verbatim:
  - `me cobraron dos veces, reembolsen o cancelo`
  - `one forward pass · any language · no API calls`
  - `36.5%` / `51.0%` / `+14.5 points from structure alone`
  - `0.84. Measured.`
  - `keep 60% · 75% accurate · escalate the rest, reason attached`

## Creative Direction
- Tone preset: polished
- Creative direction: quiet engineering flex — a demo that publishes its own error bars
- Interpretation: longer holds, fewer scenes, restrained motion; numbers carry the drama;
  editorial serif display + mono data; warm paper palette with semantic pastels only; no SaaS
  gloss, no neon, no whooshes
- Angle: every triage demo answers "which department?"; none answer "when should the model
  back off?". This one publishes the coverage/accuracy curve and picks its operating point
  from data
- Hook: a Spanish customer ticket types itself out character by character
- Outro / punchline: "At what confidence do you let a model close a ticket? / 0.84. Measured."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign

## Visual Identity
- Background: `#F7F6F3` (warm paper)
- Text: `#111111` (never pure black)
- Accents (semantic pastels only): auto `#EDF3EC`/`#346538` · escalate `#FDEBEC`/`#9F2F2D` ·
  data `#E1F3FE`/`#1F6C9F` · note `#FBF3DB`/`#956400`; hairlines `#EAEAEA`; muted `#787774`
- Display font: Newsreader (editorial serif, italic for emphasis; Google Font)
- Data font: JetBrains Mono (Google Font)
- Body font: Helvetica Neue / system sans stack
- Visual references from the project: white cards with 1px `#EAEAEA` borders and 12px radius;
  pale pastel pill tags with small uppercase mono labels; the how-it-works pipeline layout in
  docs/assets/how-it-works.html

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The ticket types itself — 3.5s — Spanish ticket typed char by char in a white card; mono
   caption `one forward pass · any language · no API calls` fades in
2. The triage assembles — 6s — department pill `payments_cash`, chip `card payment problems`,
   mono `transaction_charged_twice`, four signal rows arriving one by one (urgency 1.8 /
   frustration 1.9 / churn 0.87 / refund 0.91) with tiny bars, confidence pair cluster 0.82 ·
   intent 0.71
3. The receipts — 6s — `36.5%` muted vs `51.0%` ink with `+14.5 points from structure alone`
   counting up; then the escalation curve draws (coverage→accuracy), 0.84 threshold ticks in,
   operating point + `keep 60% · 75% accurate · escalate the rest, reason attached` with
   pale-green `auto-handle` and pale-red `escalate` pills
4. The answer — 4.5s — serif question holds, italic `0.84. Measured.` lands, mono wordmark
   `laya-triage` + `Apache 2.0 · github.com/Gjusev/laya-triage`

## Audio
- Audio role: warm bed, sparse professional accents
- Audio arc: near-silent keys under the typed hook; low warm bed rising gently through the
  assembly; one swell under the curve; quiet final hit as the number lands; fade to silence
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (~110 BPM)
- Music treatment: fade in low under Scene 1 typing (typing is the lead), rise slightly into
  Scene 3, tail fade to silence under the wordmark
- Music cue guidance: bundled preset
  `brag skill: assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`;
  beats ~0.55s apart; strong cues at 15.82s, 18.01s, 18.55s, 20.19s, 20.74s, 22.92s. Target:
  the curve reveal near a strong cue in the 15-18s region; the outro answer near 20.19s;
  signal rows (readable text) snap to every OTHER beat (~1.1s apart), not every beat
- Audio-reactive treatment: none (restraint is the aesthetic; documented by choice)
- Audio-coupled moments:
  - Scene 1 typing — subtle key ticks per character burst
  - Scene 2 signal rows — soft card sound per row arrival
  - Scene 3 threshold — one distinct tick when the 0.84 line lands
  - Scene 4 answer — one soft final hit
- SFX selection guidance: keyboard ticks, soft card/interactions, one clean tick, one soft
  impact; keep everything quieter than the reading
- SFX analysis guidance: `<skill-dir>/assets/sfx/sfx-analysis.md` (brag skill dir); prefer
  low high-frequency-risk files for the repeated card moments
- Exact SFX choice: Hyperframes chooses filenames, timestamps, density, and volume from the
  implemented animation
- Audio files: music copied to `brag-output/composition/assets/music/`; SFX to be copied by
  Hyperframes into `brag-output/composition/assets/`

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — hyperframes-core, hyperframes-
animation, hyperframes-creative, hyperframes-keyframes, hyperframes-cli. /brag is its own
workflow: do not enter the hyperframes entry-point intent interview and do not route into its
generic promo / launch-video workflow. Prefer native Hyperframes conventions over anything
in /brag.

Requirements:
- Show at least one real UI/copy/visual element from the source project (the triage card and
  the curve are the moments).
- Keep all text readable in the final render; reading floor: short labels ~0.8s settled,
  sentences ~0.3s/word.
- Keep the video at 15-25 seconds (target 20s).
- Include the music/SFX layer as planned.
- Beat locking: at most 1-3 strong-cue locks; mark `// beat-locked`; signal rows snap to
  every other beat, mark `// beat-grid`.
- Run `hyperframes check` before render — it is brag's single gate.
