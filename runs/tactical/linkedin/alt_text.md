# Alt text (English)

**Slide 1.** Dark navy hero image. On the left a price line dissolves into a cloud of noise. A dotted line leads to the right, where six glowing nodes (USA, Russia, China, Britain, India, Brazil) are linked as a network. Headline: Do whales leave signatures in technical indicators?

**Slide 2.** Schematic of the model. An OHLCV window feeds learnable indicators (14 families, 54 learnable periods, each a trainable logit; an exponential smoother is a 1-D convolution with a learned width). Then a Bi-GRU and 8-head attention, then heads for 10, 15 and 20 minutes (price change, P(up), variance). Losses: direction BCE, Gaussian NLL plus CRPS, physics-inspired terms. A dashed amber arrow shows gradients flowing back to the periods.

**Slide 3.** Forest plot of differences in direction AUC with zero marked as no difference. Learned minus textbook periods: -0.0006 [-0.006, +0.005]; periods trained by direction only: +0.0007 [-0.006, +0.008]; thin read-out learned versus frozen: -0.002 [-0.005, +0.001]; no indicators versus 14 families: -0.002, not significant. Every interval crosses zero.

**Slide 4.** Line chart of yearly direction AUC of a logistic regression, 2017 to 2025, next-open fills: 0.527, 0.562, 0.561, 0.567, 0.553, 0.540, 0.551, 0.533, 0.518, with a 95% band, a dotted close-fill line and a dashed coin-flip line at 0.50. Three cards: network 0.521 versus logistic 0.570; 0.535 versus 0.552 on five judgement folds; forward test hit rate 50.3% over 11 unseen months.

**Slide 5.** Seven translucent layers (news, GIS and terrain, elections, shareholder votes, finance and trade, social graph, physical events) stacked in perspective converge through a lens onto one thin line, the price. A callout says volatility predictability rose from 0.286 in 2018 to 0.399 in 2025 while direction skill faded.

**Slide 6.** Diagram of the hyperspace prototype. A six-node geopolitical network on a faint graticule (teal alliances, red dashed competition, amber financial flow) sits above five blocks: temporal-pattern, semantic-embedding, structural-centrality, dynamic-agent, geospatial-kernel. They feed one 80-dimensional Universal Knowledge Tensor, then SVD kernels, sparse-autoencoder concepts, and a semantic canvas with governance flags.

**Slide 7.** Closing slide: I do not know if it will work. I am optimistic. A list of planned next inputs: news, GIS layers, election results across democracies, shareholder votes across companies. A card links to github.com/another-worlds/hyperspace.

**Cover (1200x627).** Same hero image as slide 1 with the headline: Do whales leave signatures in technical indicators?
