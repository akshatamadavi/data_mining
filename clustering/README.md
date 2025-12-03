# Clustering Algorithms - Data Mining Assignment

This repository contains implementations of various clustering algorithms for the Data Mining course (CMPE-255 Sec 49). Each notebook demonstrates different clustering techniques with comprehensive documentation, visualizations, and quality metrics.

## 📋 Assignment Overview

Implementation of multiple clustering algorithms using both from-scratch implementations and state-of-the-art libraries, following assignment requirements with proper documentation, quality metrics, and visualizations.

## 📂 Repository Structure

clustering/
├── 01_Clustering.ipynb # Part A: K-Means from Scratch
├── 02_Clustering.ipynb # Part B: Hierarchical Clustering
├── 03_Clustering.ipynb # Part C: Gaussian Mixture Models
├── 04_Clustering.ipynb # Part D: DBSCAN with PyCaret
├── 05_Clustering.ipynb # Part E: Anomaly Detection with PyOD
└── README.md

## 🔬 Implementations

### Part A: K-Means Clustering from Scratch
📓 [Notebook](https://colab.research.google.com/github/akshatamadavi/data_mining/blob/main/clustering/01_CLustering.ipynb)

**Implementation Details:**
- Custom K-Means implementation without using sklearn's KMeans
- Euclidean distance calculation and centroid updates
- Convergence checking with tolerance parameter
- Tested on synthetic data (make_blobs) and Iris dataset

**Results:**
- Converged in 7 iterations
- Inertia: 140.03 (Iris dataset)
- **Quality Metrics (Iris):**
  - Silhouette Score: 0.4630
  - Davies-Bouldin Index: 0.8324
  - Calinski-Harabasz Score: 241.43

**Features:**
- Elbow method for optimal K selection
- Complete visualizations showing clustering results and centroids
- Comparison of synthetic vs real dataset performance

---

### Part B: Hierarchical Clustering
📓 [Notebook](https://colab.research.google.com/drive/1yWcxBQ-hUlnDS-jpCWBb1VDyIgIMvhkG?authuser=1)

**Implementation Details:**
- Using scipy and sklearn libraries
- Four linkage methods tested:
  - Single linkage
  - Complete linkage
  - Average linkage
  - Ward linkage

**Results:**
| Linkage Method | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
|---------------|------------------|---------------------|------------------------|
| Ward          | 0.4467          | 0.8035             | 222.72                |
| Complete      | 0.4496          | 0.7584             | 213.08                |
| Average       | 0.4803          | 0.5753             | 149.03                |
| Single        | 0.5046          | 0.4929             | 131.54                |

**Features:**
- Dendrogram visualizations for each linkage method
- Agglomerative clustering with sklearn
- Comparative analysis identifying best method
- **Key Finding:** Single linkage achieved best Silhouette score (0.5046)

---

### Part C: Gaussian Mixture Models (GMM) Clustering
📓 [Notebook](https://colab.research.google.com/drive/1x47tDTY2R_piY13LvPOHZLbe5PSxQaQO?authuser=1)

**Implementation Details:**
- Probabilistic clustering using sklearn's GaussianMixture
- Assumes data generated from mixture of Gaussian distributions
- Full covariance type with 4 components

**Results:**
- Converged in 2 iterations
- Log-likelihood: -3.71
- AIC: 2272.32
- BIC: 2357.51

**Features:**
- Hard clustering (predicted labels) visualization
- Soft clustering (probability-based) visualization with confidence opacity
- Handles overlapping clusters better than K-Means
- Model selection using AIC/BIC criteria

---

### Part D: DBSCAN Clustering with PyCaret
📓 [Notebook](https://colab.research.google.com/drive/1omdRO_rdVX2fSYiRhDICVitLX29lFGi6?authuser=1)

**Implementation Details:**
- Density-based clustering using PyCaret library
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Tested on synthetic moon-shaped clusters

**Key Features of DBSCAN:**
- Does not require specifying number of clusters beforehand
- Can identify outliers/noise in the data
- Works well with clusters of varying shapes and densities
- Requires two parameters: eps (neighborhood radius) and min_samples

**Features:**
- Non-linear cluster shape detection
- Automatic outlier identification
- Comparison with K-Means on non-spherical data

---

### Part E: Anomaly Detection using PyOD
📓 [Notebook](https://colab.research.google.com/drive/1YyqXkI8-IiblHULhByUZc3t8XoFHhTgj?authuser=1)

**Implementation Details:**
- Anomaly/outlier detection using Python Outlier Detection (PyOD)
- Multivariate dataset demonstration
- Multiple detection algorithms:
  - KNN (K-Nearest Neighbors)
  - IForest (Isolation Forest)
  - LOF (Local Outlier Factor)

**Features:**
- Comprehensive outlier detection comparison
- Visualization of anomaly scores
- Practical applications for data quality and fraud detection

---

## 📊 Clustering Quality Metrics

All implementations include comprehensive evaluation using:

1. **Silhouette Score** (Range: -1 to 1, higher is better)
   - Measures how similar an object is to its own cluster vs other clusters
   - Values near 1 indicate well-separated clusters

2. **Davies-Bouldin Index** (Lower is better)
   - Average similarity ratio of each cluster with its most similar cluster
   - Lower values indicate better cluster separation

3. **Calinski-Harabasz Score** (Higher is better)
   - Ratio of between-cluster dispersion to within-cluster dispersion
   - Also known as Variance Ratio Criterion

4. **Additional Metrics:**
   - Inertia (within-cluster sum of squares)
   - AIC/BIC for GMM model selection
   - Adjusted Rand Score for ground truth comparison

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Static visualizations and plots |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | ML algorithms, metrics, and datasets |
| **SciPy** | Hierarchical clustering and dendrograms |
| **PyCaret** | AutoML library for clustering |
| **PyOD** | Outlier detection algorithms |

---

## 🚀 Getting Started

### Prerequisites
pip install numpy pandas matplotlib seaborn scikit-learn scipy pycaret pyod


### Running the Notebooks

1. Click on any notebook link above to open in Google Colab
2. Run all cells sequentially (Runtime → Run all)
3. View visualizations and quality metrics
4. Experiment with different parameters

### Datasets Used

- **Iris Dataset:** Classic ML dataset with 150 samples, 4 features, 3 species
- **Make Blobs:** Synthetic Gaussian clusters for testing
- **Make Moons:** Non-linear crescent-shaped clusters for DBSCAN

---

## 📈 Key Features

✅ Complete implementations with detailed documentation  
✅ Quality metrics for all clustering methods  
✅ Comprehensive visualizations (scatter plots, dendrograms, elbow curves)  
✅ Comparative analysis of different methods  
✅ Both synthetic and real-world datasets  
✅ From-scratch implementations demonstrating deep understanding  
✅ State-of-the-art library usage for practical applications  
✅ Anomaly detection for outlier identification  

---

## 📖 References

### Assignment Resources
- [K-Means from Scratch Tutorial](https://colab.sandbox.google.com/github/SANTOSHMAHER/Machine-Learning-Algorithams/blob/master/K_Means_algorithm_using_Python_from_scratch_.ipynb)
- [Hierarchical Clustering with Python](https://colab.sandbox.google.com/github/saskeli/data-analysis-with-python-summer-2019/blob/master/clustering.ipynb)
- [Gaussian Mixtures - Python Data Science Handbook](https://colab.sandbox.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb)
- [PyCaret Clustering Documentation](https://pycaret.org/create-model/)
- [PyOD Documentation](https://pyod.readthedocs.io/)

### Additional Reading
- [PyCaret Clustering Tutorial](https://towardsdatascience.com/clustering-made-easy-with-pycaret-656316c0b080)
- [Anomaly Detection in Time Series](https://neptune.ai/blog/anomaly-detection-in-time-series)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)

---

## 👤 Author

**Akshata Madavi**  
San Jose State University  
CMPE-255: Data Mining  
Fall 2025

---

## 📝 Assignment Notes

- All notebooks include detailed explanations and code comments
- Step-by-step implementations demonstrate understanding of algorithms
- Proper clustering quality measures included as required
- Visualizations enhance understanding of clustering behavior
- Each part submitted as separate Colab notebook with complete documentation

---

## 🎯 Learning Outcomes

Through this assignment, I gained hands-on experience with:

1. **Algorithm Implementation:** Building K-Means from scratch reinforced understanding of centroid-based clustering
2. **Hierarchical Methods:** Comparing linkage methods revealed trade-offs in cluster quality
3. **Probabilistic Models:** GMM demonstrated soft clustering and model selection techniques
4. **Density-Based Clustering:** DBSCAN showed superiority for non-spherical clusters
5. **Anomaly Detection:** PyOD provided practical tools for outlier identification
6. **Evaluation Metrics:** Comprehensive understanding of clustering quality measures

---

## 📄 License

This project is part of academic coursework for CMPE-255 at San Jose State University.

---

**GitHub Repository:** [akshatamadavi/data_mining/clustering](https://github.com/akshatamadavi/data_mining/tree/main/clustering)
