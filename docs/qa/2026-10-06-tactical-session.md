# Owner answers, 2026-10-06: a separate session for tactical experiments

The owner's words are quoted verbatim (Russian, translated in brackets).

**Announcement.** Owner: "У тебя будет отдельная от работы по MVP сессия заточенная на тактические эксперименты
с нейросетью" [You will have a session, separate from the MVP work, focused on tactical experiments with the
neural network.]

The lead asked four questions (one round, with recommendations). The answers:

1. **Where the session keeps code and results.** Options: its own worktree and branch (recommended); directly in
   `remediation/plan` under `runs/tactical/`; CPU analysis only, no code. Owner: its own worktree and branch.
2. **How the GPU is shared.** Options: a lock file, first come first served (recommended); tactical priority;
   MVP priority. Owner: **MVP priority** (not the recommended option).
3. **Rigour and budget.** Options: fast, no pre-registration, about 3 GPU-hours a day without asking, defaults
   change only through the MVP backlog and D-025 (recommended); as the MVP studies (SPEC, 5 judgement folds,
   3-hour cap per study); no GPU limit. Owner: the recommended option.
4. **Who runs training.** Options: the tactical lead itself (recommended, for speed); as in the MVP
   (implementer for code, experimenter for the GPU). Owner: **as in the MVP** (not the recommended option).

Recorded as D-062; the rules are in [../TACTICAL.md](../TACTICAL.md).

## Round 2: the hill-climb on output accuracy

Owner: "я могу тебе дать hillclmb по метрикам точности выхода" [I can give you a hill-climb on output accuracy
metrics]. The lead asked four questions. Answers:

1. **Metric:** direction AUC, mean of h0-h2 (recommended); CRPSS and conformal coverage are guard-rails.
2. **Levers:** everything except the fixed decisions (not the recommended Config-only option).
3. **Folds:** climb on 5 dev folds, check the winner once on 5 other dev folds (recommended).
4. **Budget (free text, verbatim):** "переписываю прошлое правило: используем GPU параллельно с другой сессией.
   Бюджет - некотролируем. правило: обучение на сверхкоротких массивах. Даю 2 минуты максимум на каждый прогон"
   [rewriting the previous rule: the GPU in parallel with the other session; budget uncontrolled; training on
   ultra-short arrays; 2 minutes at most per run]. D-063.

**Goal (verbatim, same round):** "Цель - попытка выйти из стратегической ловушки оптимизацией поиском тактического
прорыва в расчете нейрокни. Риск менеджмент - отдельный независимый бранч" [The goal: an attempt to escape the
strategic trap by optimisation, searching for a tactical breakthrough in the network's computation. Risk
management is a separate, independent branch.]
