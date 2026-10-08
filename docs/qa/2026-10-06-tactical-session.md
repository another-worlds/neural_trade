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

## Round 3 (2026-10-07): design, candidate, heads

The lead measured the error of smaller designs on the 975 tactical runs and reported the 0.8 AUC values (one block,
the Easter weekend of 2022-04-16, where a fade-10-bars rule scores 0.756). Owner answers, verbatim:

- Screening tolerances: "для нас удовлетворительна лшибка 0.05, а ложные победы - 10%" [an error of 0.05 and 10%
  false wins are acceptable for us]; then "П.2 6х2 берем" [point 2: we take 6x2].
- The 7-day scale-up: "Пункт 1. я разрешаю" [I allow it]; then "Возьми пока 1 кандидата и его проверь, у которого
  наивысший скор 0.8 или больше" [for now take 1 candidate, the one with the highest score, 0.8 or more, and check it].
  The 10-candidate run was stopped (default, candidates 1-2 complete: no gain; 3-5 one slice each).
- Heads: "Добавь в измерение остальные 8 голов" [add the other 8 heads to the measurement]; then "Головы - отлично.
  на будущее используй их всегда" [the heads: great; use them always from now on].
- "Да, проверь срез. Разметь его рамки и если можешь распарси новости" [check the slice, mark its bounds, parse the
  news if you can]: runs/tactical/slice_2022_04_19.py / .json; journal H7.
- "Формализуй требования, дай фидбек после" [formalise the requirements, give feedback afterwards]:
  runs/tactical/cand_0805/SPEC.md.
- Dashboard: "обнови также наш HTML. и при каждом новом прогоне добавляй активную вкладку в HTML, чтобы я всегда мог
  отследить наш прогресс и статус без дергания тебя" [update our HTML too, and with every new run add an active tab, so I
  can always follow our progress and status without pinging you]. TACTICAL.md "One dashboard tab per run".
- "Проверь гипотезу с эпохами" [check the epochs hypothesis]: runs/tactical/epochs_1d/SPEC.md.
- Aggregated hill-climb metric (asked 2026-10-07, two options each): "3 группы поровну, по шуму (Recommended)" and
  "1 день, если догонит (Recommended)": price, direction and confidence weigh 1/3 each, every paired difference scaled by
  its seed noise (runs/tactical/hc4_metric.py); the hill-climb runs on the 1-day block if the epochs check shows it
  reaches 7-day quality, otherwise on 7 days.
- After "стоп по всем задачам останови трейнинг и инференс" [stop all tasks, stop training and inference] (done): "Продолжи.
  Не контролируй чтобы твои процессы конкретно по процессрору и ОЗУ не приводили к критической загрузке" [continue; (do)
  control that your processes do not lead to a critical CPU and RAM load] - read as "control" (the lead said so and asked
  to be corrected otherwise). TACTICAL.md "CPU and RAM guard".

## Round 4 (2026-10-08): architecture, heads, plan

- "Идеально. обязательно пиши в план, я об этом уже забыл" [write it into the plan] - the geometry-over-indicators idea:
  runs/tactical/PLAN.md section 1 (the tactical plan file, kept from now on).
- "Запустим тактический эксперимент где полностью удалим эту голову из лоссов и архитектуры ... я хочу сравнить 6 голов
  ансамбль против 2 голов" [remove the price head from losses and architecture; compare the 6-head ensemble with 2 heads];
  clarified by question: "3 горизонта vs 1 горизонт" (no-price network on 3 horizons vs on 1). PLAN section 2.
- "Также еще учти уверенность. Выдай если мы повышаем порог уверенности ... насколько ... прибавляем точность" [show how
  accuracy grows with the confidence threshold]: journal H15.
- Proposals 1-3 (combination backtest on CPU, the two-heads round, new code for geometry and gating): "да" to all.
- 2026-10-08, after WSL went down: the lead stopped its GPU jobs and proposed a stricter rule while Docker/WSL runs (1 process,
  4 GB GPU memory). Owner (verbatim): "можно. на докер не обращай внимания , закрой правило" [go ahead; ignore Docker; close the
  rule]: no Docker-specific rule; the existing load guard stays.
- 2026-10-08: "почему так долго ... совсем не тактические сроки" [why so slow; not tactical timescales] -> fast probe (1 run, 3
  epochs, every 20 steps). "Какие приоритеты есть в винде ... давай пересмотрим" [which Windows priorities exist; let's
  revisit]: asked with options; owner chose "Ниже среднего (Recommended)" (below normal) for the lead's training processes.
