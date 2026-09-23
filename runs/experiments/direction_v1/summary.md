# Direction experiments - development folds

AUC with the 5 bps deadband mask on each development fold's own test block, model / logistic regression
on trailing returns fit on the same fold's train block. Fold -1 (the reported test block) is not used.

| experiment | epochs | AUC h0 | AUC h1 | AUC h2 | MCC h1 | P(up) std h1 | delta corr h1 | min | run |
|---|---|---|---|---|---|---|---|---|---|
| `bce_f-2` | 20 | 0.4958 / 0.4973 | 0.4940 / 0.4916 | 0.4943 / 0.4990 | -0.0001 | 0.0098 | +0.0157 | 5 | `20260923T170844Z-493b803-5312581c-bce_f-2` |
| `bce_f-3` | 13 | 0.5063 / 0.5348 | 0.5004 / 0.5440 | 0.4839 / 0.5205 | +0.0000 | 0.0015 | -0.0179 | 4 | `20260923T165416Z-1124317-dirty-ed810f56-bce_f-3` |
| `bce_no_soft_ece_f-3` | 12 | 0.4941 / 0.5348 | 0.4973 / 0.5440 | 0.4481 / 0.5205 | +0.0379 | 0.0290 | +0.0539 | 4 | `20260923T165752Z-6f02b1a-dirty-1e411657-bce_no_soft_ece_f-3` |
| `bce_skip_f-2` | 13 | 0.4754 / 0.4973 | 0.5208 / 0.4916 | 0.5205 / 0.4990 | +0.0323 | 0.0934 | +0.0094 | 7 | `20260923T170111Z-0fae28e-29f98ea5-bce_skip_f-2` |
| `bce_skip_f-3` | 20 | 0.5356 / 0.5348 | 0.5147 / 0.5440 | 0.5158 / 0.5205 | +0.0284 | 0.1030 | +0.0520 | 7 | `20260923T170111Z-0fae28e-babed164-bce_skip_f-3` |
| `bce_skip_l2_1e-2_f-2` | 20 | 0.4806 / 0.4973 | 0.5485 / 0.4916 | 0.4539 / 0.4990 | +0.0788 | 0.0989 | +0.0087 | 5 | `20260923T171811Z-493b803-dirty-2db11242-bce_skip_l2_1e-2_f-2` |
| `bce_skip_l2_1e-2_f-3` | 20 | 0.5506 / 0.5348 | 0.5006 / 0.5440 | 0.5184 / 0.5205 | +0.0027 | 0.1022 | +0.0621 | 4 | `20260923T171411Z-493b803-dirty-d7b2fc0e-bce_skip_l2_1e-2_f-3` |
| `legacy_focal_dice_f-3` | 9 | 0.4851 / 0.5348 | 0.4950 / 0.5440 | 0.4899 / 0.5205 | -0.0256 | 0.0037 | -0.0019 | 3 | `20260923T165416Z-1124317-dirty-f2647353-legacy_focal_dice_f-3` |
