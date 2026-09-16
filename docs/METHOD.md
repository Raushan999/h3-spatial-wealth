# Method: full technical walkthrough

This document is the complete, end-to-end technical explanation of the project: what data goes in, how it is transformed, how the graph neural network is built and trained, how a final rupee value is produced for every hexagon, and how the results are (and are not) validated.

The [README](../README.md) gives the short, plain-language version. This document gives the full version, including every mathematical expression used in the pipeline, worked numerical examples, the reasoning behind each design choice, and a question-and-answer section addressing the most common points of confusion or challenge.

---

## Table of contents

1. [Overview of the full pipeline](#1-overview-of-the-full-pipeline)
2. [Administrative data and the target quantity](#2-administrative-data-and-the-target-quantity)
3. [Building the H3 hexagon grid](#3-building-the-h3-hexagon-grid)
4. [Features attached to every hexagon](#4-features-attached-to-every-hexagon)
5. [Feature preprocessing](#5-feature-preprocessing)
6. [The graph: nodes, edges, and hierarchy](#6-the-graph-nodes-edges-and-hierarchy)
7. [Why a graph neural network](#7-why-a-graph-neural-network)
8. [The neural network architecture](#8-the-neural-network-architecture)
9. [Training the network](#9-training-the-network)
10. [From raw score to final rupee value](#10-from-raw-score-to-final-rupee-value)
11. [The wealth index](#11-the-wealth-index)
12. [What is learned vs. what is fixed arithmetic](#12-what-is-learned-vs-what-is-fixed-arithmetic)
13. [Validation and evaluation](#13-validation-and-evaluation)
14. [Known limitations](#14-known-limitations)
15. [Alternative methods considered](#15-alternative-methods-considered)
16. [Technology stack](#16-technology-stack)
17. [Questions and answers](#17-questions-and-answers)

---

## 1. Overview of the full pipeline

The project converts one published number per district into a spatial surface of estimated values, following this sequence:

```
Official district GDP (one number per district)
        +
District boundary shapes
        ↓
Fill each district with ~5 km² hexagonal cells (H3 grid)
        ↓
Attach observed data to every cell: population, night lights,
roads, buildings, land use, points of interest
        ↓
Transform raw values into model-ready numeric features
        ↓
Connect cells into a graph: neighbouring cells + zoomed-out parent cells
        ↓
Train a graph neural network so that, when cell outputs are summed
per district, they approximate the official district GDP
        ↓
Rescale each district's cell outputs so they sum EXACTLY to the
official GDP (a fixed arithmetic correction, not a learned step)
        ↓
Convert final cell GDP into per-person GDP and a 0–100 percentile
("wealth index") within the state
        ↓
Export cell-level results as CSV / JSON
        ↓
A static web map reads these exported files and renders the hexagons
```

Every stage above is described in full mathematical detail in the sections that follow.

The training and prediction stages run once per state, offline, producing a data file. The web map that end users interact with never runs the neural network itself — it only displays numbers that were already computed.

---

## 2. Administrative data and the target quantity

### 2.1 What is observed

For each state, the pipeline uses one official table of **district-level GDP** (referred to here as Gross District Domestic Product, GDDP), published by that state's economics and statistics department. This is the **only** monetary ground truth used anywhere in the project.

For a district $g$, denote its official GDP as:

$$
Y_g
$$

This is a single scalar value, in rupees, per district, per year. It is treated as exactly correct and is never adjusted by the model.

### 2.2 What is not observed

There is no dataset, anywhere used by this project, that measures GDP, income, or output at a scale smaller than one district. Every value produced below the district level in this project is a **model-based allocation**, not an independent measurement.

### 2.3 Matching districts across data sources

District names differ across data sources — administrative boundary files, GDP tables, and historical naming conventions do not always agree (for example, a district may be spelled differently, split, merged, or renamed over time). Before any modelling happens, district names from the boundary data and the GDP table are reconciled into one consistent set of names per state, and in a few cases, multiple administrative units are combined into a single reporting unit to match how the official GDP figure was published (for example, when a state publishes one shared total for two administratively separate but jointly reported districts).

---

## 3. Building the H3 hexagon grid

### 3.1 What H3 is

H3 is an open hierarchical hexagonal grid system that divides the surface of the earth into cells at multiple resolutions. At a chosen resolution, every cell is a roughly-regular hexagon of similar area.

This project uses:

- **Resolution 7** — approximately 5 km² per cell — as the unit of final prediction.
- **Resolution 6** and **Resolution 5** — each covering a progressively larger area — as *parent* cells used to give the model a zoomed-out view of a broader neighbourhood.

Every resolution-7 cell has exactly one resolution-6 parent, and every resolution-6 cell has exactly one resolution-5 parent, forming a fixed nesting structure:

$$
\text{cell}_{R7} \subset \text{cell}_{R6} \subset \text{cell}_{R5}
$$

### 3.2 Filling a district with hexagons

Given a district's boundary polygon, the H3 grid is used to enumerate every resolution-7 cell whose centre point falls inside that polygon. Each retained cell is recorded together with its resolution-6 parent, its resolution-5 parent, its centroid coordinates, and its surface area.

For a district $g$ this produces a set of cells:

$$
\{1, 2, \dots, N_g\} \quad \text{where } N_g \text{ is the number of resolution-7 cells in district } g
$$

A cell is assigned to a district based on whether its centre point lies inside that district — a cell is not split or area-weighted across a boundary even if part of its hexagon physically extends outside the district.

### 3.3 Why hexagons rather than administrative sub-units (wards, villages)

Villages, wards, and other administrative sub-units vary enormously in size and shape, which makes them poor units for consistent spatial comparison and for defining a regular neighbourhood structure. A hexagonal grid instead gives:

- roughly equal cell area everywhere in a state,
- a fixed number of touching neighbours for every interior cell (six),
- a natural nesting into progressively larger cells, useful for representing broader spatial context.

---

## 4. Features attached to every hexagon

For every resolution-7 cell, the following raw quantities are computed and attached:

| Feature family | What it captures |
|---|---|
| Population | Number of people estimated to live inside the cell |
| Night-time lights | Mean night-time radiance (brightness) inside the cell |
| Road length | Total length of major roads passing through the cell |
| Points of interest (POIs), by category | Counts of restaurants, cafés, bars, hospitals, banks, malls, supermarkets, hotels, offices, leisure venues, and similar categories |
| Building footprint area and count | Total built-up floor footprint and number of distinct buildings |
| Land-use shares | Fraction of the cell's area classified as residential, commercial, industrial, or retail |

None of these features are monetary. Each is treated as a **proxy signal** — a piece of indirect evidence that may correlate with the intensity of local economic activity, with well-understood weaknesses:

- Night lights primarily reflect electrified, night-visible activity; agricultural or informal daytime activity does not necessarily show up.
- Road and point-of-interest data come from a community-maintained open map, whose completeness tends to be higher in urban areas than in rural ones.
- Building footprint area does not capture building height, so a single-storey warehouse and a multi-storey office block can appear similar.
- Land-use classification reflects mapped land parcels, which again tend to be more completely mapped in urban areas.

Population and the district GDP figure are also on different timelines from the satellite and map layers, since each source is refreshed on its own schedule. This is a genuine limitation and is treated as such throughout this document.

---

## 5. Feature preprocessing

Raw features are not fed directly into the neural network. Two transformations are applied.

### 5.1 Zero-inflation split

Many features (for example, the count of hospitals in a cell) are zero for a large fraction of cells and only become meaningful once they are non-zero. To represent this cleanly, each raw magnitude feature $x$ is converted into two separate signals:

**An existence indicator:**

$$
x_{\text{exists}} =
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x = 0
\end{cases}
$$

**A log-magnitude, defined only where the value is positive:**

$$
x_{\text{log}} =
\begin{cases}
\log_{10}(x) & \text{if } x > 0 \\
0 & \text{if } x = 0
\end{cases}
$$

This split matters because $\log_{10}(1) = 0$ and an undefined $\log_{10}(0)$ would otherwise have to be replaced with an arbitrary value — without the existence indicator, "zero of something" and "exactly one of something" would look identical to the model.

### 5.2 Standardisation (z-scoring)

After the log/exists split, every resulting numeric column $c$ is standardised across all cells in the state, using that column's own mean $\mu_c$ and standard deviation $\sigma_c$:

$$
c_{z} = \frac{c - \mu_c}{\sigma_c}
$$

This puts every feature on a comparable numeric scale (roughly centred at 0 with unit spread), which is important for stable neural network training — without it, features with naturally larger raw magnitudes (such as building area in square metres) would dominate features with naturally smaller raw magnitudes (such as a 0/1 existence flag) purely due to scale, not actual importance.

Share-type features (the land-use fractions, which are already bounded between 0 and 1) skip the log/exists split and are z-scored directly.

After this stage, each resolution-7 cell is represented by a fixed-length numeric vector of standardised features. This is the vector that the neural network actually receives as input for that cell.

---

## 6. The graph: nodes, edges, and hierarchy

### 6.1 Nodes

Three sets of nodes exist in the graph for a given state:

- **Resolution-7 nodes** — one per hexagon; these are the only nodes for which a final prediction is produced.
- **Resolution-6 nodes** — one per unique resolution-6 parent of the resolution-7 cells present.
- **Resolution-5 nodes** — one per unique resolution-5 parent of the resolution-6 cells present.

Resolution-6 and resolution-5 nodes do not have independently observed features. Their feature vector is the **average** of the standardised feature vectors of their child cells at the level below:

$$
x^{(6)}_p = \frac{1}{|C(p)|} \sum_{i \,\in\, C(p)} x^{(7)}_i
$$

where $C(p)$ is the set of resolution-7 cells whose parent is $p$. The same averaging rule is applied one level up, from resolution-6 to resolution-5.

### 6.2 Same-level edges (adjacency)

At each resolution, an edge is created between two cells of that resolution if they are immediate neighbours on the hexagonal grid (share a border). For an interior cell this typically produces six neighbours; cells at the edge of a state or district may have fewer. These edges are undirected — information can flow in either direction along them — and a cell is never connected to itself.

### 6.3 Cross-level edges (parent–child)

An edge is created from every resolution-7 cell to its resolution-6 parent, and from every resolution-6 cell to its resolution-5 parent. These edges allow information to move both **upward** (child cells informing their parent) and **downward** (a parent cell informing its children) during the network's computation.

### 6.4 What an edge represents

An edge is a channel along which the model can combine *numeric representations* of two cells. It does **not** represent money, population, or any physical quantity moving between locations. No rupee is transferred from one hexagon to another; the graph exists purely so the model can let a cell's estimate be informed by the character of its surroundings, at more than one spatial scale.

---

## 7. Why a graph neural network

### 7.1 The core motivation

A model that only looks at a cell's own features (population, lights, roads, etc.) cannot represent the idea that "this cell sits right next to a major commercial hub" or "this cell is surrounded by farmland." Two cells with identical own-features but very different surroundings would be forced to receive identical predictions under a features-only model.

A graph neural network is used specifically to let a cell's prediction be influenced by:

1. its own features,
2. the features (or, after several rounds, the *learned representations*) of its immediate neighbours,
3. a broader, zoomed-out summary of the region around it, via the resolution-6 and resolution-5 parent cells.

### 7.2 What a graph neural network is, conceptually

A graph neural network is a model designed to operate on data structured as a network of connected nodes rather than a flat table of independent rows. It repeatedly performs two operations at every node:

1. **Aggregation** — combine information from a node's connected neighbours into a single summary.
2. **Update** — combine that summary with the node's own current representation to produce a new, refined representation.

Repeating this process for several rounds ("layers") lets information from increasingly distant parts of the graph reach a given node, indirectly, through its neighbours' neighbours.

### 7.3 What is explicitly not being modelled

The graph does not represent a causal or physical flow of economic value between hexagons. It is a computational mechanism for **sharing evidence**, not a claim about how the economy actually operates spatially. This distinction matters: a high output for a cell is not "money flowing in from next door" — it is the network concluding, based on that cell's and its surroundings' combined evidence, that the cell is likely to be relatively more economically intense.

---

## 8. The neural network architecture

### 8.1 Input embedding

Each node's standardised feature vector is first passed through a separate linear transformation (one per resolution level — resolution 7, 6, and 5 each have their own), followed by a non-linear activation function (ELU, the Exponential Linear Unit), producing an initial internal representation for that node:

$$
h_i^{(0)} = \text{ELU}\left(W_{\text{in}} \, x_i + b_{\text{in}}\right)
$$

Here $x_i$ is the node's standardised feature vector, $W_{\text{in}}$ and $b_{\text{in}}$ are learned parameters (weights and bias), and $h_i^{(0)}$ is the resulting internal representation — a fixed-length vector of numbers with no direct real-world unit, used purely internally by the network.

### 8.2 Same-level message passing (GraphSAGE-style)

At each layer $\ell$, every node aggregates information from its same-level neighbours by averaging their current representations:

$$
m_i^{(\ell)} = \underset{j \,\in\, N(i)}{\text{mean}} \; h_j^{(\ell)}
$$

where $N(i)$ is the set of neighbours of node $i$ at that resolution level. The node then combines this neighbour summary with its own current representation, through two separate learned linear transformations, followed by a non-linearity:

$$
a_i^{(\ell)} = \text{ELU}\left(W_{\text{self}}^{(\ell)} \, h_i^{(\ell)} \;+\; W_{\text{nb}}^{(\ell)} \, m_i^{(\ell)}\right)
$$

$W_{\text{self}}^{(\ell)}$ controls how much weight the node's own current representation gets; $W_{\text{nb}}^{(\ell)}$ controls how much weight the aggregated neighbour representation gets. Both are learned during training, separately for each resolution level and each layer.

**Worked numerical example** (using a single number per node, instead of a full vector, purely to illustrate the arithmetic):

Suppose a cell's current representation is $h_i = 4$, and its three neighbours currently have representations $2$, $5$, and $8$. The neighbour aggregation is their mean:

$$
m_i = \frac{2 + 5 + 8}{3} = 5
$$

If the learned weights happen to be $W_{\text{self}} = 0.7$ and $W_{\text{nb}} = 0.3$:

$$
a_i = \text{ELU}\big(0.7 \times 4 \;+\; 0.3 \times 5\big) = \text{ELU}(4.3) \approx 4.3
$$

In the real network this happens with 64-dimensional vectors and full weight matrices rather than single numbers, but the underlying arithmetic — a weighted combination of "self" and "neighbour average" — is the same.

### 8.3 Upward pooling (children → parent)

At every layer, each resolution-6 node also receives an aggregated summary of its resolution-7 children's current representations (a simple average), and each resolution-5 node receives an aggregated summary of its resolution-6 children:

$$
u_p^{(\ell)} = \frac{1}{|C(p)|} \sum_{i \,\in\, C(p)} h_i^{(\ell)}
$$

### 8.4 Downward broadcasting (parent → children)

Each parent node's current representation is also passed back down to its children, through a learned linear transformation:

$$
d_i^{(\ell)} = W_{\text{down}}^{(\ell)} \, h_{\text{parent}(i)}^{(\ell)}
$$

This is how a resolution-7 cell receives a signal that reflects the character of its broader resolution-6 (and, indirectly through further layers, resolution-5) surroundings — a form of context far wider than its immediate six neighbours.

### 8.5 Combining everything into an updated representation

For a resolution-7 node, the same-level neighbour-aggregated signal and the downward parent signal are combined (concatenated) and passed through one more learned linear layer and a non-linearity, producing the updated representation for that layer:

$$
h_{i,\text{new}}^{(\ell+1)} = \text{ELU}\Big(W_{\text{mix}} \big[\, a_i^{(\ell)} \,\Vert\, d_i^{(\ell)} \,\big]\Big)
$$

where $\Vert$ denotes concatenating the two vectors together before the linear transformation. Resolution-6 and resolution-5 nodes are updated analogously, combining their own same-level aggregation with the relevant upward and/or downward signals available to them.

### 8.6 Initial residual connection

After many rounds of neighbour-averaging, node representations across a densely connected graph tend to become increasingly similar to one another — a well-known effect called **oversmoothing**, where the network progressively loses the ability to tell nodes apart. To counteract this, a small fraction of each node's *original* (layer-0) representation is mixed back in at every layer:

$$
h_i^{(\ell+1)} = (1 - \alpha) \, h_{i,\text{new}}^{(\ell+1)} + \alpha \, h_i^{(0)}
$$

with $\alpha$ a small constant (set to $0.1$ in this project). This keeps a persistent trace of each cell's own original identity present throughout the network, even after several rounds of neighbourhood mixing.

### 8.7 Number of layers and this process repeated

This entire update (same-level aggregation, upward pooling, downward broadcasting, mixing, residual) is repeated for a fixed number of layers — four, in this project. After four layers, a resolution-7 cell's representation has, in principle, been influenced by information originating from its own features, its immediate neighbours, its resolution-6 parent's other children (siblings), and a wider resolution-5 region, since each additional layer lets information travel one further step through the graph.

### 8.8 Final output layer

After the last layer, each resolution-7 cell's final representation is passed through one more learned linear layer, producing a single number, which is then passed through a **softplus** activation function to guarantee it is non-negative:

$$
s_i = \text{softplus}\big(W_{\text{out}} \, h_i^{(L)} + b_{\text{out}}\big) = \ln\!\big(1 + e^{\,W_{\text{out}} h_i^{(L)} + b_{\text{out}}}\big)
$$

$s_i$ is the **raw economic intensity score** for cell $i$ — a single non-negative number per hexagon. It has no rupee value or real-world unit on its own; its scale only becomes meaningful once used in training, described next.

---

## 9. Training the network

### 9.1 What is being learned

The learnable parameters of the network are every weight matrix and bias vector introduced above: the three resolution-specific input transformations, the self/neighbour transformations used in same-level aggregation at every layer, the transformations used for downward broadcasting at every layer, the layer-mixing transformation, and the final output layer. Nothing about the H3 grid, the graph's edges, the raw feature values, the district assignment, or the official GDP figures is learned — those are all fixed inputs.

### 9.2 Turning cell scores into a district-level prediction

For every district $g$, the model's predicted total is obtained by weighting each cell's raw score by its population and summing over all cells that belong to that district:

$$
\widehat{Y}_g = \sum_{i \,:\, d(i) = g} s_i \cdot \text{population}_i
$$

where $d(i)$ denotes the district that cell $i$ belongs to.

**Worked numerical example.** Suppose a (very small, illustrative) district contains three cells:

| Cell | Population | Raw score $s_i$ | Contribution $s_i \times \text{population}_i$ |
|---|---:|---:|---:|
| A | 100 | 10 | 1,000 |
| B | 200 | 20 | 4,000 |
| C | 300 | 10 | 3,000 |

$$
\widehat{Y}_g = 1{,}000 + 4{,}000 + 3{,}000 = 8{,}000
$$

If the official GDP for this district is $Y_g = 10{,}000$, the model's raw, pre-correction prediction under-estimates the true total by 2,000.

### 9.3 The loss function

Training compares the model's predicted district totals against the official district totals, using a **mean squared error computed on a logarithmic scale**:

$$
L_{\text{agg}} = \frac{1}{|G|} \sum_{g \,\in\, G} \Big(\log_{10}(\widehat{Y}_g + \varepsilon) \;-\; \log_{10}(Y_g + \varepsilon)\Big)^2
$$

where $G$ is the set of districts included in training, $|G|$ is how many there are, and $\varepsilon$ (a very small constant, $10^{-6}$) simply avoids taking the logarithm of exactly zero.

**Why a logarithmic scale.** Without the logarithm, an error of a fixed rupee amount in a very large district (say, a state capital with a huge economy) would dominate the loss compared to the same *proportional* error in a much smaller district. Using logarithms means that under-predicting a district's GDP by, say, 20% contributes a similar amount to the loss whether that district's economy is very large or very small — the loss is sensitive to **relative**, not absolute, error.

**Worked numerical example**, continuing from above:

$$
L_{\text{agg}} = \Big(\log_{10}(8{,}000) - \log_{10}(10{,}000)\Big)^2 = (3.9031 - 4.0000)^2 \approx 0.0094
$$

### 9.4 A secondary regularising term

A second, smaller term is added to discourage the model from predicting extremely low, near-zero intensity scores, weighted by each cell's population share of the state's total population:

$$
L_{\text{floor}} = \frac{\displaystyle\sum_i \text{population}_i \cdot \big[\max(0,\; \tau - s_i)\big]^2}{\displaystyle\sum_i \text{population}_i}
$$

with $\tau$ a small fixed threshold. This term contributes nothing for any cell whose score already exceeds $\tau$, and only penalises cells whose score falls below it, roughly in proportion to how populated that cell is.

### 9.5 Combined loss

The two terms are combined with a fixed small weight on the regularising term:

$$
L = L_{\text{agg}} + 0.05 \cdot L_{\text{floor}}
$$

This single combined value is what the training process tries to minimise.

### 9.6 Backpropagation and parameter updates

Training proceeds by standard neural network optimisation:

1. **Forward pass** — compute $s_i$ for every cell, then $\widehat{Y}_g$ for every district, then the loss $L$.
2. **Backward pass (backpropagation)** — compute the gradient of the loss with respect to every learnable parameter $\theta$, written $\nabla_\theta L$. This tells the training process how a small change in each parameter would change the overall loss.
3. **Parameter update** — adjust every parameter a small step in the direction that reduces the loss:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \, \nabla_\theta L
$$

where $\eta$ is the learning rate, a small positive number controlling the step size.

4. **Repeat** — this cycle is repeated for many rounds ("epochs"), each time processing the entire state's graph at once (all cells, all districts, in a single pass — not split into smaller batches).

The optimiser used to perform this parameter update is **Adam**, a widely used adaptive optimisation algorithm, with an initial learning rate of $0.01$ that is gradually reduced over the course of training following a cosine schedule. Gradients are clipped to a maximum size before each update, a standard technique to prevent unstable, overly large parameter jumps early in training.

### 9.7 Why the loss cannot pinpoint individual cells

Because $L_{\text{agg}}$ only ever compares a **summed** district total against the official figure, an error signal like "this district's total is too low" says nothing about *which* individual cell within that district is responsible, or by how much. The gradient distributes a correction across every cell in that district, proportionally to how each cell's score currently contributes to the sum and how the shared network parameters connect that cell's inputs to its output. Since the same set of parameters is used to compute the score for every cell in the state, the optimisation process is really searching for **one shared rule** (a function of features and graph position) that, applied everywhere, produces district sums close to every district's official figure simultaneously — not a separate, independent answer for each of the many thousands of individual cells.

This is an instance of what is sometimes called **weak** or **aggregate supervision**: the model is trained using labels that exist only at a coarser level than the individual predictions it produces.

### 9.8 Why rupee-scale values emerge from lights, roads, and population alone

The raw features (night-light brightness, road length, population, etc.) carry no inherent monetary unit. Before training, an untrained network's output $s_i$ is essentially arbitrary in scale. It is **only** because the loss function repeatedly compares $\sum_i s_i \times \text{population}_i$ against a real rupee figure, epoch after epoch, that the optimisation process is pushed toward parameter values which make $s_i$ land on a scale where this sum is close to the true district GDP. This is the same underlying idea used whenever any machine learning model is trained to predict a monetary quantity from non-monetary input features (for example, predicting a house price from its number of rooms and floor area): the scale of the output is entirely determined by what the model is trained against, not by any inherent property of the inputs.

### 9.9 Repeating training with multiple random initialisations

Because the starting values of all learnable parameters are randomly initialised, different runs of training (started from different random seeds) can converge to different, similarly-well-fitting solutions — this is expected behaviour in neural network optimisation, especially for a problem with this much freedom in how the district-level constraint can be satisfied internally (see Section 14). To reflect this, training is repeated independently five times, using five different random seeds. For every cell, the five resulting raw scores are averaged to produce a single central estimate, and the minimum and maximum across the five runs are also recorded and exported, to show how sensitive that particular cell's score is to the random starting point.

This minimum–maximum range reflects **optimisation variability**, not a statistically calibrated confidence interval or margin of error.

---

## 10. From raw score to final rupee value

### 10.1 The scaling problem

After training, a district's predicted total $\widehat{Y}_g$ will typically be **close** to, but not exactly equal to, the official figure $Y_g$ — training minimises an average error across many districts simultaneously, so no single district is guaranteed a perfect fit.

### 10.2 The dasymetric correction factor

To guarantee that every district's cells sum to *exactly* the correct official total, one purely arithmetic correction factor is computed per district, **after** training is complete, with no further learning involved:

$$
k_g = \frac{Y_g}{\displaystyle\sum_{i \,\in\, g} s_i \times \text{population}_i}
$$

This single number answers: *by what factor must the model's raw output for this district be multiplied so that it matches the true total exactly?*

### 10.3 Final per-cell GDP

Each cell's final allocated GDP is then:

$$
GDP_i = s_i \times \text{population}_i \times k_g
$$

Since $k_g$ is identical for every cell within a given district, it rescales all of that district's cells by the same proportion — it does not change how the district's GDP is *distributed* between cells, only the absolute magnitude of every cell's share, all together:

$$
\sum_{i \,\in\, g} GDP_i = \sum_{i \,\in\, g} s_i \times \text{population}_i \times k_g = k_g \sum_{i \,\in\, g} s_i \times \text{population}_i = k_g \times \widehat{Y}_g = Y_g
$$

This equality holds **exactly**, by construction, for every district, every time. It is not a measure of how good the underlying model is — it is a guaranteed consequence of the formula.

### 10.4 Rewriting the formula as a share

Substituting the definition of $k_g$ into the formula for $GDP_i$ gives an equivalent, and perhaps clearer, way to see the result:

$$
GDP_i = Y_g \times \frac{s_i \times \text{population}_i}{\displaystyle\sum_{j \,\in\, g} s_j \times \text{population}_j}
$$

In words: **each cell receives the district's official GDP, multiplied by that cell's share of the district's total weighted score.** The neural network's only real contribution is deciding these relative shares; the absolute rupee total per district is always, by construction, the officially published figure.

### 10.5 Worked numerical example

Returning to the earlier three-cell district example, where the raw predicted total was $\widehat{Y}_g = 8{,}000$ against an official total of $Y_g = 10{,}000$:

$$
k_g = \frac{10{,}000}{8{,}000} = 1.25
$$

$$
GDP_A = 1{,}000 \times 1.25 = 1{,}250 \qquad
GDP_B = 4{,}000 \times 1.25 = 5{,}000 \qquad
GDP_C = 3{,}000 \times 1.25 = 3{,}750
$$

$$
1{,}250 + 5{,}000 + 3{,}750 = 10{,}000 \;\checkmark
$$

The relative shares are preserved exactly: cell A keeps 12.5% of the district's total, cell B keeps 50%, and cell C keeps 37.5%, whether or not the scaling correction is applied.

### 10.6 GDP per person

Dividing a cell's final GDP by its own population gives its estimated GDP per capita:

$$
\text{GDP per capita}_i = \frac{GDP_i}{\text{population}_i} = s_i \times k_g \qquad (\text{for } \text{population}_i > 0)
$$

Note that population **cancels out** of this per-capita figure — it only ever entered the calculation through the district total and each cell's own weighting, not as an independent multiplier on the final per-capita number. Two cells within the same district that happen to receive the same raw score $s_i$ will always end up with the *same* GDP-per-capita figure, regardless of how different their populations are; what differs between them is their *total* GDP contribution, not their *per-person* figure.

---

## 11. The wealth index

After every cell's GDP-per-capita figure is computed, all *inhabited* cells within one state are ranked from lowest to highest GDP-per-capita, and each cell is assigned its **percentile position** in that ranking, producing a number between 0 and 100:

$$
\text{wealth\_index}_i = \text{percentile rank of GDP per capita}_i \text{ among inhabited cells in the same state}
$$

A cell with a wealth index of, for example, 38 sits above roughly 38% of the state's inhabited cells (by this project's own estimated GDP-per-capita figure) and below roughly 62% of them.

This value is **not** derived from any household survey, census wealth index, or asset-ownership data. It is purely a within-state rank of this project's own model-derived, allocation-based GDP-per-capita estimate. It should not be interpreted as, or compared against, independently collected household wealth or living-standards measures.

---

## 12. What is learned vs. what is fixed arithmetic

| Component | Learned by the network | Fixed, non-learned calculation |
|---|---|---|
| How features, neighbours, and parent context combine | ✓ | |
| The raw intensity score $s_i$ for every cell | ✓ | |
| The district correction factor $k_g$ | | ✓ (computed once, after training, per district) |
| Final cell GDP $GDP_i$ | | ✓ (arithmetic: $s_i \times \text{population}_i \times k_g$) |
| The guarantee that a district's cells sum to $Y_g$ | | ✓ (a direct algebraic consequence of how $k_g$ is defined) |
| The GDP-per-capita percentile ("wealth index") | | ✓ (a ranking operation over already-computed values) |

Only the *relative pattern within* a district — which cells are estimated to be more or less intense than others — comes from the trained network. The *absolute total* for each district is always, and only, the officially published figure.

---

## 13. Validation and evaluation

### 13.1 The fundamental identifiability problem

For a district with $N_g$ cells, there is exactly **one** real equation available to check any candidate allocation against:

$$
\sum_{i=1}^{N_g} GDP_i = Y_g
$$

This single equation is satisfied by an enormous number of different possible ways of splitting $Y_g$ across $N_g$ cells — far more unknowns ($N_g$ individual cell values) than equations (1 per district). Nothing about the features, the graph structure, or the training procedure changes this basic fact: **there is no way, using only district-level totals, to prove that one particular internal split is the uniquely correct one**, because the true fine-scale split has never been independently measured anywhere in the data used here. The features and the graph structure only influence *which* plausible split the model converges to — they narrow the search using reasonable assumptions, but they cannot verify the result against ground truth that does not exist in this dataset.

This is an example of what is sometimes called an **ecological inference** problem: inferring something about individual units (hexagons) from information that is only available in aggregate (districts).

### 13.2 Checks that are actually performed

Given the above, the following diagnostic checks are computed and reported for every trained state, each with a specific and limited interpretation:

**Reconstruction check.** After the dasymetric correction, the sum of every district's cell GDP is compared back against the official figure. This is expected to match almost exactly (differences on the order of floating-point rounding error), because it is a direct algebraic consequence of the correction formula in Section 10 — **this check confirms the arithmetic was implemented correctly, and says nothing about the quality of the underlying spatial allocation.**

**Pre-scaling district fit.** The training loss itself (Section 9.3), evaluated on the raw, uncorrected model output, measures how closely the network's own predictions — before the exact correction is applied — track the official district figures. This reflects how well the network fits the districts it was trained on, but says nothing about accuracy *within* any individual district.

**Spatial hold-out.** A subset of districts (roughly one-fifth) is excluded from contributing to the training loss, and the network is retrained; the raw, pre-correction predicted totals for these held-out districts are then compared against their official figures, using mean absolute percentage error. This is intended as a check of how well the model generalises to districts it was not directly fitted against. In practice this hold-out has real limitations: the held-out districts' cells remain physically present in the graph (so information can still reach them through neighbouring or parent cells that *were* used in training), and the feature standardisation statistics (Section 5.2) are computed across the whole state, including the held-out districts, before the split is made — both of which allow some information leakage from the held-out districts into the training process, meaning this check is a useful but imperfect signal of generalisation, not a fully clean test.

**Feature ablation.** The model is trained once with, and once without, the night-time-lights feature, and the resulting pre-scaling district fit is compared between the two. A meaningfully worse fit without night lights indicates that this feature is contributing useful signal toward matching district totals — but again, this only speaks to district-level fit, not cell-level accuracy.

**Rank-correlation sanity checks.** The final wealth index is compared, using rank correlation, against a few of the same raw input signals used to train the model (night lights, road length, point-of-interest counts). A positive and meaningful correlation is an expected, mild sanity check — since these very features were used as model inputs, a correlation between them and the model's output is not independent evidence of accuracy, only a check that the output has not become disconnected from the inputs it was trained on.

### 13.3 What these checks do not establish

None of the checks above involve an independently measured fine-scale (sub-district) economic quantity. As a result, **this project does not have direct evidence that its hexagon-level GDP allocations are individually accurate.** What it does have good evidence for is that (a) the arithmetic correctly preserves official district totals, and (b) the raw model reasonably approximates district-level totals, including to a meaningful — if imperfect — degree on districts excluded from training.

Any future validation against genuinely independent, finer-than-district data (for example, geo-located household survey data, sub-district administrative statistics, or firm/establishment-level records not used anywhere in training) would materially strengthen or weaken confidence in the fine-scale allocation, and none of that currently exists in this pipeline.

---

## 14. Known limitations

- **No fine-scale ground truth.** The central and most important limitation: nothing in the training data measures true GDP below the district level, so the internal spatial pattern cannot be independently verified (Section 13).
- **Temporal mismatch.** The official GDP figure, the population estimate, the night-lights composite, and the map data (roads/buildings/POIs) are each drawn from different reference years or update schedules, and are not adjusted to a common reference point.
- **Price-basis mismatch across states.** Some states publish GDP in constant (inflation-adjusted) prices and others in current prices; absolute rupee values are not directly comparable across such states without further adjustment.
- **Uneven map data completeness.** Road, building, and point-of-interest data from the open community map tends to be more complete in urban areas than in rural ones, which can bias features (and therefore outputs) toward appearing more "active" in well-mapped areas independent of true economic activity.
- **Night-light bias.** Night-time brightness favours electrified, visible, night-time activity and can under-represent agriculture and informal daytime economic activity.
- **Population used twice.** Population is both an input feature to the network and the direct multiplier used to weight each cell's contribution to the district sum — a structural choice worth being explicit about, since it means population indirectly influences the outcome through two separate channels.
- **Adjacency is a simplification.** Physical neighbouring cells are assumed to be the most relevant spatial context, but real economic relationships (for example, along a transport corridor connecting two non-adjacent hexagons) may not align with simple physical adjacency.
- **Oversmoothing risk.** Repeated neighbour-averaging across several layers can, in principle, make nearby cells' outputs converge toward each other more than is warranted; the residual connection (Section 8.6) mitigates but does not eliminate this risk.
- **Cell assignment by centroid.** A hexagon is assigned entirely to one district based on where its centre point falls, even though part of its physical area may fall just outside that district's boundary.
- **Forced allocation.** Every rupee of a district's official GDP is always distributed somewhere among that district's cells, even in areas where the available features and evidence are thin or of low quality — the total must go *somewhere*, whether or not the available signal is strong enough to place it confidently.
- **Single, undifferentiated correction unit for very large administrative mergers.** Where several administrative units are combined into a single official reporting figure, the model has essentially no way to differentiate GDP *between* those merged units — only *within* the merged, combined area as a whole.
- **Minimum–maximum range across random seeds is not a statistical confidence interval.** It reflects sensitivity to random initialisation during optimisation, not a calibrated measure of prediction uncertainty.

---

## 15. Alternative methods considered

| Approach | How it would work | Strengths | Weaknesses relative to this project's needs |
|---|---|---|---|
| **Population-proportional allocation** | $GDP_i = Y_g \times \dfrac{\text{population}_i}{\sum_j \text{population}_j}$ | Extremely simple, transparent, requires no training | Assumes every person contributes identically to GDP regardless of location; ignores all non-population evidence and all spatial context |
| **Linear regression** | A single weighted sum of standardised features feeding the same district-level loss | Simple, interpretable, fast to train | Cannot represent non-linear feature interactions or make use of neighbouring-cell context |
| **Random Forest** | Tree-based ensemble model | Captures non-linear relationships and feature interactions well | Standard implementations expect a label for every individual training row; here, no individual-cell label exists, so it would need substantial adaptation to work with only district-level totals, and does not naturally incorporate neighbour or multi-scale spatial context |
| **Gradient-boosted trees (e.g. XGBoost)** | Similar to Random Forest, typically stronger predictive performance on tabular data | Same non-linear strength as Random Forest, generally very competitive on tabular problems | Same district-level-label adaptation issue as Random Forest; no built-in mechanism for neighbour or multi-scale context |
| **Gradient-boosted trees with hand-built spatial features** | As above, but with manually engineered features summarising neighbouring cells (e.g. average night lights of the six nearest hexagons) | Can approximate much of the spatial-context benefit of a graph model, while remaining simpler and more interpretable | Requires manually deciding which spatial summaries to construct, rather than letting the model learn this; still needs the same district-level-label adaptation |
| **Graph neural network (used here)** | As described throughout this document | Learns which neighbouring and multi-scale context matters, rather than requiring it to be manually specified; naturally suited to data with an inherent spatial network structure | Considerably more complex to implement, train, and interpret than the alternatives above; higher risk of oversmoothing; and — importantly — this project does not currently include a direct, controlled comparison against the simpler alternatives above, so it cannot be concluded from the available evidence that the added complexity produces materially better real-world allocations |

The graph neural network is a reasonable and principled choice given the goal of incorporating spatial context without hand-engineering it, but its necessity relative to the simpler alternatives above — particularly gradient-boosted trees with hand-built spatial features — has not been directly demonstrated within this project, since no side-by-side comparison against those alternatives has been run.

---

## 16. Technology stack

| Layer | Technology used | Role |
|---|---|---|
| Numerical / graph computation | Python, PyTorch | Defining and training the neural network, computing gradients, running the optimiser |
| Spatial grid | H3 (hexagonal hierarchical spatial index) | Dividing districts into resolution-7/6/5 cells and defining adjacency/parent relationships |
| Geospatial processing | Shapely, GeoPandas-style boundary handling | Reading and intersecting administrative boundary polygons with the hexagon grid |
| Raster / satellite data handling | rasterio | Reading and sampling the night-time-lights satellite raster data |
| Open map data extraction | OpenStreetMap raw extracts, processed directly | Deriving road length, building footprints, land-use shares, and point-of-interest counts per cell |
| Tabular data handling | pandas, NumPy | Feature engineering, aggregation, joining data sources, exporting results |
| Optimiser | Adam (with cosine learning-rate scheduling and gradient clipping) | Updating network parameters during training |
| Model architecture components | Linear layers, ELU activation, softplus activation, GraphSAGE-style neighbour aggregation, hierarchical pooling/broadcasting, GCNII-style initial residual connections | Building blocks of the graph neural network described in Section 8 |
| Output formats | CSV, compressed CSV, JSON | Exporting per-cell results for downstream use |
| Interactive visualisation | Static HTML/JavaScript web page, MapLibre for map rendering, H3's JavaScript library for drawing hexagons client-side | Displaying exported results as an interactive map, with no server-side computation or live model inference involved |

The interactive map is a **static** front end: it loads pre-computed result files and renders them; it does not invoke the neural network live, and there is no prediction API that computes a new value on demand.

---

## 17. Questions and answers

**What exactly does the model predict?**
The model predicts a single non-negative number per hexagonal cell, called the raw economic intensity score. It does not directly predict a rupee value; the rupee value is obtained afterward by multiplying this score by the cell's population and applying the fixed district-level correction factor described in Section 10.

**If we never have true GDP for any individual hexagon, how can this be trained at all?**
Training uses a form of aggregate (or "weak") supervision: individual cell scores are summed, weighted by population, into a district total, and that total — not any individual cell value — is compared against the one real number that does exist, the official district GDP. The model is only ever graded on how well these summed totals match reality, never on any individual cell's value.

**Why does a district's cells always sum exactly to the official GDP figure? Does that prove the model is accurate?**
They sum exactly because of a deliberate, separate arithmetic step (Section 10) applied after training: every cell's raw output is multiplied by one shared correction factor per district, specifically chosen so the sum comes out exactly right. This is a guaranteed mathematical consequence of that formula, not evidence that the underlying spatial pattern within the district is correct. See Section 13.1 for why this cannot be independently verified with the data currently available.

**Why use a graph neural network instead of feeding each cell's own features into a standard model?**
Because a model that only sees a cell's own features cannot distinguish between two cells with identical own-features but very different surroundings (one next to a commercial hub, one surrounded by farmland). A graph neural network lets each cell's prediction be informed by its neighbours' and its broader region's characteristics, which a features-only model structurally cannot do.

**Could a simpler method (e.g. splitting GDP purely by population, or a gradient-boosted tree model) work just as well?**
Possibly, and this has not been ruled out. Population-proportional allocation is a strong, transparent baseline that this project does not directly benchmark against. Gradient-boosted trees with manually engineered spatial-context features (for example, average neighbouring night-light brightness) could plausibly capture much of the same benefit as the graph neural network, with less complexity. No controlled comparison between the graph neural network and these simpler alternatives currently exists within this project (see Section 15), so a claim that the graph neural network is necessary, or superior, cannot currently be supported by the available evidence.

**Does the graph literally move GDP or money between neighbouring hexagons?**
No. The graph only allows the model to combine internal numeric representations between connected cells during computation. No rupee amount, population count, or other physical quantity is transferred between cells; only internal, unit-less representations used by the network are shared.

**Two cells have identical population but different lights, roads, and points of interest — will they receive the same final GDP?**
Not necessarily. Their raw intensity scores can differ because those scores depend on each cell's own features and its neighbourhood, not on population alone. Since final GDP is $s_i \times \text{population}_i \times k_g$, two equally-populated cells with different scores will receive different total GDP figures — though if their scores happen to be equal, their GDP-per-capita figures will be identical regardless of any other feature differences (Section 10.6).

**Two cells have identical own-features (same lights, roads, POIs, population) but sit in different parts of a district — could they still get different scores?**
Yes, because each cell's neighbours and its broader resolution-6/resolution-5 context are also part of the input to the network. Two otherwise-identical cells with different surroundings can receive different scores.

**What are resolution-6 and resolution-5 cells for, if the model only outputs a value at resolution-7?**
They exist purely to give the network a computationally efficient way to represent a broader, zoomed-out summary of an area than a single hexagon's six immediate neighbours could provide. A resolution-7 cell can be influenced by a wide surrounding region within a small number of processing layers, using this hierarchy, rather than needing very many rounds of purely local neighbour-to-neighbour passing to reach the same distance.

**Are resolution-6 and resolution-5 nodes independently measured?**
No. Their feature vectors are computed as the plain average of their children's already-standardised feature vectors (Section 6.1). They carry no independent data source of their own.

**Is the district or the state itself a node the network reasons about directly?**
No. Districts and states organise which cells' training loss and final rescaling get grouped together, but there is no dedicated "district node" or "state node" inside the graph or the network. The only nodes the network processes are hexagonal cells at resolutions 7, 6, and 5.

**Why is the training loss computed on a logarithmic scale rather than plain rupee error?**
Because districts vary enormously in economic size. Without a logarithmic scale, errors in the very largest districts would dominate the overall training signal, and the model would effectively be trained mostly to fit a handful of the biggest economies while barely being penalised for getting much smaller districts wrong in relative terms. The logarithmic scale makes the loss sensitive to proportional (percentage-like) error rather than absolute rupee error, so districts of every size contribute comparably.

**Why does population appear both as an input feature to the network and as a separate multiplier in the final formula?**
Population serves two distinct roles: as an input feature, it lets the network learn how population level itself might relate to relative economic intensity (for example, very sparsely populated cells may systematically differ from densely populated ones in ways beyond what other features capture); as a separate multiplier, it converts a per-person intensity score into a population-weighted total contribution to the district sum, which is necessary because two cells with the same intensity but very different populations should contribute very differently to a district's total GDP. This dual role is a deliberate design choice, though it does mean population's influence on the outcome flows through the model in more than one way, which is worth being explicit about when interpreting results.

**Since night-time lights, road data, and points of interest are all inputs to the model, does it mean anything that the model's output correlates with those same inputs afterward?**
Only in a limited sense. Such a correlation confirms that the model's output has not become disconnected from the very features it was trained on, which is a basic sanity check. It is not independent evidence of accuracy, because the same signals were already available to, and used by, the model during training — a correlation between an input and an output derived largely from that input is expected, not a separate confirmation of correctness.

**How is the raw model output "corrected" into a final rupee figure, mathematically?**
A single factor per district, $k_g = Y_g / \sum_{i \in g} s_i \times \text{population}_i$, is computed once after training. Multiplying every cell's raw weighted score in that district by this one factor makes the district's cells sum to exactly the official total, while leaving every cell's relative *share* of that total unchanged (Sections 10.2–10.4).

**If a district's correction factor is far from 1 (for example, around 0.4), what does that indicate?**
It indicates that the raw, uncorrected network output for that particular district was substantially off from the official total — in the example above, roughly two-and-a-half times too high — before the fixed correction was applied. A correction factor far from 1 is a useful diagnostic signal that the raw model fit that specific district poorly, even though the final, corrected output for that district will still sum exactly to the official figure regardless.

**Does a district's correction factor say anything about how that district compares to other districts?**
No. Each district's correction factor is calculated entirely independently, using only that district's own official total and that district's own raw predicted total. It has no role in comparing the relative importance, size, or wealth of different districts to one another — it exists purely to correct that one district's internal arithmetic.

**Is `wealth_index` the same thing as a household wealth or living-standards score?**
No. It is a percentile ranking, within one state, of this project's own model-derived, allocation-based GDP-per-capita estimate. It is not derived from, calibrated against, or validated using any independent household survey, asset-ownership data, or living-standards measurement.

**What would it take to know whether the hexagon-level estimates are actually accurate?**
An independent source of genuinely fine-scale (sub-district) economic data that was not used anywhere in training — for example, geo-located household survey results, sub-district or block-level administrative economic statistics, or establishment/firm-level records — compared directly against the model's cell-level output. No such comparison currently exists in this project; the checks that do exist (Section 13) operate only at the district level or check internal consistency, neither of which can confirm fine-scale accuracy.

**Is there a live prediction service that computes a new estimate on demand?**
No. All computation (data collection, feature engineering, graph construction, network training, and the final rescaling) happens offline, once per state, and produces exported result files. The interactive map is a static web page that only reads and displays those already-computed files; it does not run the neural network itself.

**Does training happen jointly across all states at once, or separately per state?**
Separately, per state. A distinct network is trained for each state, using only that state's own districts, cells, and features. This avoids one state's economic patterns directly influencing another state's allocation, though it also means that a state with very few districts has correspondingly very little aggregate signal available to train against.
