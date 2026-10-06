# Owner quiz, 2026-10-03

Seven decisions that were still open in STATUS "Waiting for the owner", question 8, and NT-103.
The owner picked one option on each. The strings below are the option labels as selected.
Questions 7 and 9, the C: copy, pushing, and the window-free plan were already answered and were
not asked again.

**Question 8, a new data source (taker-buy volume first).** Selected: "No new source".
No Binance taker-buy volume, basis, or funding. The micro loop closes on its journal and the lead
returns to the MVP backlog. D-050.

**NT-007, which delta the strategies read.** Selected: "Split raw and served (Recommended)".
Coherence checks (`magnitude_coherent`, `direction_aligned`) use the raw heads. Position size stays
on the shrunk served delta. The enhanced_multi_horizon `d1 > 0` entry rule was not part of this
option, so it stays on the served delta. D-051. Code is NT-115.

**NT-006, physics-term re-run.** Selected: "Approve the re-run (Recommended)".
About 7-10 GPU-hours, the estimate asked on 2026-09-25, over the 3-hour cap. The run itself is not
started by this answer. D-052.

**NT-103, epoch selection.** Selected: "Add the switch (Recommended)".
`EPOCH_SELECT_METRIC` may be added. The served epoch stays the best validation-loss epoch until a
later paired test. D-053.

**NT-008, merge into master.** Selected: "Not yet (Recommended)".
`master` stays untouched. Nightly stays off until a later yes. D-054.

**Inference latency.** Selected: "File the P3 item (Recommended)".
Filed as NT-116. Not started. D-055.

**NT-017, licence.** Selected: "MIT".
`LICENSE` is the MIT licence, copyright 2026 another-world. README points at it. QA has not passed
the file. D-056.
