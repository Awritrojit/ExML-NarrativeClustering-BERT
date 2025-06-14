# Project ExML: Exploring Localization in BERT Neural Activations

## Overview
This project investigates the localization of stimulus processing in large language models (LLMs), specifically BERT, by analyzing and visualizing neural activations across its layers. The study leverages dimensionality reduction, clustering, and group discriminative metrics to understand how BERT internally represents language stimuli from different authors and novels.

## Key Features
- **Activation Analysis:** Loads and processes BERT activation data for various novels, authors, and text variants.
- **Dimensionality Reduction:** Applies PCA and MDS to project high-dimensional activations into 2D for visualization.
- **Clustering & Metrics:** Uses clustering and computes the Group Discriminative Value (GDV) to quantify group separability in the reduced space.
- **Visualization:** Generates interactive and static plots to illustrate trends and localization of neural activations across BERT layers.
- **Synthetic Data:** Supports analysis on both real and synthetic (neural-style-transfer) data to validate findings.

## Code Structure
- `main.py`, `old_main.py`: Main analysis and visualization scripts.
- `Code/`: Contains modules for data generation, BERT processing, GDV computation, and utility functions.
- `Data/`: Stores activation matrices, dimensionality reduction results, and metric CSVs.
- `infographics/`: Output plots and visualizations.
- `Trial1/`: Additional data, experiments, and presentation materials.

## Methodology
1. **Data Preparation:** Extract BERT activations for each layer, author, and novel.
2. **Matrix Conversion:** Convert nested activation dictionaries to matrices for analysis.
3. **Dimensionality Reduction:** Apply PCA and MDS to visualize activations in 2D.
4. **Localization & Clustering:** Visualize and analyze how activations cluster by author or novel.
5. **Metric Computation:** Calculate GDV to assess group separability across layers.
6. **Trend Analysis:** Plot GDV trends to observe how localization evolves through BERT's layers.

## Usage
1. Place activation data in the `Data/` directory.
2. Run `main.py` to generate visualizations and metrics.
3. View results in the `infographics/` directory.

## Conclusion
Our initial hypothesis that localization of stimulus processing occurs even for sequential stimuli—in this case, language—has been confirmed in the LLM BERT. BERT, being biologically plausible as per [1]-[6], makes it suitable for investigative studies in Cognitive Computational Neuroscience. The validity of the hypothesis, even with classical analytical methods on synthetic data generated by neural-style-transfer, demonstrates the strength of the hypothesis and the applicability of artificial neural networks as candidates for understanding biological neural functions. Read more [here](https://arxiv.org/abs/2501.08053).

**References:**
1. N. Kriegeskorte and P. K. Douglas, “Cognitive computational neuroscience,” Nature Neuroscience, vol. 21, no. 9, pp. 1148–1160, 2018.
2. A. Schilling, W. Sedley, R. Gerum, C. Metzner, K. Tziridis, A. Maier, H. Schulze, F.-G. Zeng, K. J. Friston, and P. Krauss, “Predictive coding and stochastic resonance as fundamental principles of auditory phantom perception,” Brain, vol. 146, no. 12, pp. 4809–4825, 2023.
3. P. Stoewer, C. Schlieker, A. Schilling, C. Metzner, A. Maier, and P. Krauss, “Neural network based successor representations to form cognitive maps of space and language,” Scientific Reports, vol. 12, no. 1, p. 11233, 2022.
4. P. Stoewer, A. Schilling, A. Maier, and P. Krauss, “Neural network based formation of cognitive maps of semantic spaces and the putative emergence of abstract concepts,” Scientific Reports, vol. 13, no. 1, p. 3644, 2023.
5. “Multi-modal cognitive maps based on neural networks trained on successor representations,” arXiv preprint, 2023.
6. K. Surendra, A. Schilling, P. Stoewer, A. Maier, and P. Krauss, “Word class representations spontaneously emerge in a deep neural network trained on next word prediction,” arXiv preprint, 2023.
