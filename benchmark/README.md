# Turkish Decision Benchmark

`tr_decision_bench_v1.jsonl` contains 2,194 realistic Turkish messages with about 20,000 labeled decision questions.
It is intended for measuring the performance of decision models (Decision 3.0, Pusula or others) on Turkish.
The benchmark is also published on Hugging Face as [erenextlabs/turkish-decision-benchmark](https://huggingface.co/datasets/erenextlabs/turkish-decision-benchmark).

The benchmark is developed by Erenext's AI Lab.

- **Coverage:** all 20 domains of the Pusula taxonomy and its 299 intents are represented; 12 writing styles are included (formal e-mails,
  everyday chat, no Turkish characters and typos, angry capitals, long and rambling, one short line, Turkish–English
  code-switching, an elderly user's roundabout politeness, young and emoji-heavy, several issues with the real request
  last, error codes and logs, voice dictation without punctuation); about 10% security probes (prompt injection,
  jailbreak, unauthorized access, harmful requests, spam); fake personal data in about 30%.
- **Urgency reflects the situation, not the tone:** the word "acil" appears in 696 messages, 55% of which are routine
  or can wait a few days.
- **Questions per message:** `alan` / `alt_alan` / `niyet` (domain → subdomain → intent routing), the six universal
  questions (`aciliyet`, `eylem_gerekli`, `insana_devret`, `guvenlik`, `duygu`, `kisisel_veri`) and, for some messages,
  an ad-hoc business question with its own options.
- **Labels:** every question was answered blind and independently by two annotators; disagreements were resolved by
  a third, more careful pass. All labels follow a single written [labeling policy](../pusula/data/label_policy.md),
  which defines the boundaries of the judgment questions (when urgency is high, when a case should be handed off to a
  person, and when an action is required). Annotator agreement before adjudication, which indicates the ambiguity of
  each question, is shown below:

  | question | agreement |
  |---|---|
  | alan / alt_alan / niyet | 97.5 / 96.6 / 94.6 |
  | aciliyet | 85.2 |
  | eylem_gerekli | 92.7 |
  | insana_devret | 87.2 |
  | guvenlik | 99.2 |
  | duygu | 90.2 |
  | kisisel_veri | 96.8 |

- **Construction:** messages were written by Claude (Anthropic) from randomized recipes (intent × style × length ×
  situation) and labeled with Claude models for evaluation purposes only; near-duplicates were removed. The benchmark
  contains no real people, companies or personal data. Messages that the annotation model declined to label (a small
  number of harmful-content probes) are not included.

## Results

Macro accuracy (%) on this benchmark (19,861 questions): the accuracy of each of the 10 question types is computed
separately, and the mean of these accuracies is reported.

| Model | Accuracy |
|---|---|
| Decision 3.0 (`vllm-sr/d3`) | 88.5 |
| Pusula 1.0 | 92.6 |
| **Pusula 2.0** | **92.0** |
| Pusula Nano 1.0 | 82.7 |
| **Pusula Nano 2.0** | **83.2** |

- The per-question (micro) accuracy is lower, because the 115 custom questions (`ozel_soru`), which are answered
  almost without error, carry less weight: 87.5 for Decision 3.0 and 91.4 for Pusula 2.0.
- The Pusula 2.0 score follows the default release behavior (`PUSULA_OWN_BASE`): the custom questions are answered by
  the base model with the adapter disabled. With the adapter enabled for every question, as in the evaluation command
  below, Pusula 2.0 scores 92.2.
- The Decision 3.0 and Pusula scores were measured in BF16.

## Format

One JSON object per line, in the Pusula / d3 request format:

```json
{
  "id": "…",
  "state": "Turkish message",
  "questions": {"alan": {"type": "choice", "instructions": "…", "criteria": {"…": "…"}}, "…": {}},
  "labels": {"gold": {"alan": "finans_bankacilik", "aciliyet": "2", "eylem_gerekli": "true", "…": "…"}}
}
```

Each question is a complete d3 `systemone` question, so any d3-compatible model can be scored by sending
`{"state", "questions"}` and comparing its answers with `labels.gold`
(`choice` → option key, `noul` → `"true"`/`"false"`, `score` → level index as a string).

```bash
python -m pusula.d3train eval --d3 ~/d3 --data benchmark/tr_decision_bench_v1.jsonl --out base_d3.json
```

`--preds base_d3_preds.jsonl` additionally writes one line per question in benchmark order (message id, question,
option keys, gold key, predicted key and the probability of each option).

This benchmark is intended for evaluation only and should not be used for training.
