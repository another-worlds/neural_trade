# Owner answers, 2026-10-06: the takeover reconciliation and the way of working

The owner's words are quoted verbatim (Russian, translated in brackets where the meaning goes beyond the
offered options).

**Reconciliation plan.** The lead reported the stall of 2026-10-01, the takeover session's branch `nt-099`,
the folder move into `D:/nt/` and the QA audit, and proposed a four-step order (environment, rescue the
uncommitted work, merge with corrections, QA and continue).
Owner: "Да, напиши план по своему порядку." [Yes, write the plan in your order.] The plan was approved in plan
mode, including the correction that the shipped `LAMBDA_VOL: 0` stays (it reproduces the tested 0.1 floor
under calibration) and the owner-approved env change (the editable install re-pointed to `D:\nt`). D-058.

**Slow suite.** Asked why waiting and token cost grew several-fold, the lead offered three cuts (CI lint plus
fast only; drop nightly; slow suite only before merging large changes). Owner: "3". D-059.

**QA by risk and test tiers.** Asked how effective QA is ("равные части имплементора и ка это абсурдно или
нет" [equal shares of implementer and QA: is that absurd or not]), the lead proposed five changes. Owner: "да
запиши эти пункты. оптимизируй также условия запуска тестов чтобы они не тормозили разработку." [yes, record
these points; also optimise when tests run so they do not slow development down.] D-060.

**Models and effort.** Owner: "Я хочу чтобы на планах и на лиде ты использовал opus 5.5 high. При исследованиях
ultracode. Я хочу чтобы ты сейчас в целом исходя из доков и опыта индустрии выяснил какую схему эффортов и
моделей лучше всего сделать сейчас. Должен быть какой-то стандарт примерно." [On plans and the lead use Opus
5.5 high; for research, ultracode; find out from the docs and industry experience the best scheme of efforts
and models now; there should be a standard.] Then: "найди способ применить эту схему к проекту. Эта схема должна
быть автоматически всегда использована при любой работе в проекте. Используй этих агентов, чтобы у нас с тобой
был пункт управления, а ты сам менеджил их. Также скажи с какого момента войдет в силу данная идея" [apply it
to the project, automatically and always; use these agents so that we have a control point and you manage
them; say from when it takes effect.] D-061.
