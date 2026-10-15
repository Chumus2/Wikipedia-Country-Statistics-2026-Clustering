# 🌍 Global Country Demographics & Geography Clustering Project

## 1. Project Overview
This project applies unsupervised machine learning techniques to segment the world's countries based on comprehensive geographic and demographic features (including land area, inland water, population density, HDI, fertility rate, land per capita, and development indices). The primary goal was to uncover hidden structural patterns, group similar nations, and isolate global outliers using advanced density-based clustering.

---

## 2. Pipeline and Methodology
* **Data Preprocessing & EDA:** Cleaned raw statistics for 195 countries, handling missing values and feature scaling.
* **Dimensionality Reduction (PCA):** Transformed high-dimensional features into principal components (`PCA1`, `PCA2`) to capture maximum variance and visualize data distribution.
* **Hyperparameter Tuning (Optuna):** Optimized clustering algorithms (K-means, DBSCAN, HDBSCAN) by tracking comparative quality metrics (Silhouette Score, Davies-Bouldin, Calinski-Harabasz).
* **Final Segmentation & Profiling:** Merged cluster labels with original data, mapped numeric identifiers to country names, and profiled each group by its mean demographic traits.

---

## 3.Final Clustering Results & Profiling
* **Noise / Outliers (39 countries):**
    * *Key members:* Russia, Canada, China, United States, Australia, India, Singapore, Monaco, Vatican City, etc.
    * *Characteristics:* Global superpowers, massive geographic entities, or micro-states whose extreme parameters caused them to be isolated from standard groups by density algorithms.
* **Cluster 0 — Major Global Baseline & Core Regions (126 countries):**
    * *Key Members:* Brazil, Germany, France, Japan, Mexico, UK, Poland, Argentina, South Africa, etc.
    * *Characteristics:* The core demographic and economic baseline representing the majority of standard nation-states across Europe, Latin America, and parts of Asia/Africa.
* **Cluster 1 — Developing Compact Regions (6 countries):**
    * *Key Members:* Lesotho, Equatorial Guinea, Djibouti, Timor-Leste, Vanuatu, Samoa.
    * *Characteristics:* Small-to-medium developing states sharing specific transitional population distribution traits.
* **Cluster 2 — High-Income Energy & Industrial Hubs (4 countries):**
    * *Key Members:* Kuwait, Qatar, Cyprus, Trinidad and Tobago.
    * *Characteristics:* High-income economies driven by specialized industrial or energy structures.
* **Cluster 3 — Vulnerable Transitional States (6 countries):**
    * *Key Members:* Palestine, Comoros, São Tomé and Príncipe, Kiribati, Tonga, Micronesia.
    * *Characteristics:* Small island or geographically constrained regions facing unique socio-economic vulnerabilities.
* **Cluster 4 — Advanced Micro-Economies & Islands (3 countries):**
    * *Key Members:* Luxembourg, Mauritius, Saint Lucia.
    * *Characteristics:* High-density specialized financial, trade, or tourism hubs with distinct demographic advantages.
* **Cluster 5 — Dense Island Enclaves & Micro-States (11 countries):**
    * *Key Members:* Bahrain, Dominica, Palau, Seychelles, Maldives, Liechtenstein, etc.
    * *Characteristics:* A tightly knit cluster characterized by exceptionally high population densities relative to their tiny landmasses.

---

## 4. Key Takeaways and Next Steps
* **Macro-Geographic Realism:** Density-based clustering successfully separated standard nations from spatial outliers
(such as continental giants and city-states).
* **Saved Artifacts:**
    * Processed data & final assignments stored in `final_df.csv`
    * Cluster demographic profiles exported to `cluster_profiles.csv`
    * Model optimization metrics tracked in `model-comparison.csv`
