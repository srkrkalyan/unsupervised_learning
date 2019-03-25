Unsupervised Learning and Dimensionality Reduction

Complete source code used to reproduce experiments used in writing analysis report of unsupervised and dimensionality reduction algorithms is available at below GitHub repo:

https://github.com/srkrkalyan/unsupervised_learning.git

In order to reproduce either of the clustering or dimensionality reduction or ANN on clustering + dimensionality reduction experiments, please follow below mentioned steps: 

1. Clone or download all source files and data sets from below GitHub repo into one local directory:

https://github.com/srkrkalyan/unsupervised_learning.git

2. System requirements before running any .py files downloaded in step #1:

	a. Python 3.6.8
	b. matplotlib 3.0.2
	c. sklearn 0.20.2
	d. pandas 0.24.0
	e. numpy 1.15.4
	f. scipy 1.2.0


3. Install Python 3.6 environment and other modules listed in step #2

4. Provided below is the file name and associated experiment:

	Clustering Experiments:

	a. breast_cancer_k_means.py: k-means clustering, print metrics, plot clustering results of Breast Cancer Dataset
	b. breast_cancer_EM.py: EM clustering, print metrics, plot clustering results of Breast Cancer Dataset
	c. travel_insurance_k_means.py: k-means clustering, print metrics, plot clustering results of Travel Insurance 	   Dataset
	d. travel_insurance_EM.py: EM clustering, print metrics, plot clustering results of Travel Insurance Dataset


	Dimensionality Reduction Experiments:
	
	e. breast_cancer_PCA.py: PCA transformation, runs ANN on PCA transformed dataset, k-means & EM clustering of PCA transformed breast cancer dataset
	f. breast_cancer_ICA.py: ICA transformation, runs ANN on ICA transformed dataset, k-means & EM clustering of ICA transformed breast cancer dataset
	g. breast_cancer_Random Projections.py: RP transformation, runs ANN on RP transformed dataset, k-means & EM clustering of RP transformed breast cancer dataset
	h. breast_cancer_SVD.py: SVD transformation, runs ANN on SVD transformed dataset, k-means & EM clustering of SVD transformed breast cancer dataset
	i. travel_insurance_PCA.py: PCA transformation, runs ANN on PCA transformed dataset, k-means & EM clustering of PCA transformed dataset, runs ANN on data projected using clustering algorithms for travel insurance dataset
	j. travel_insurance_ICA.py: ICA transformation, runs ANN on ICA transformed dataset, k-means & EM clustering of ICA transformed dataset, runs ANN on data projected using clustering algorithms for travel insurance dataset
	k. travel_insurance_Randomized Projections.py: RP transformation, runs ANN on RP transformed dataset, k-means & EM clustering of RP transformed dataset, runs ANN on data projected using clustering algorithms for travel insurance dataset
	l. travel_insurance_SVD.py: SVD transformation, runs ANN on SVD transformed dataset, k-means & EM clustering of SVD transformed dataset, runs ANN on data projected using clustering algorithms for travel insurance dataset


	Datasets:

	m. breast_cancer.data.csv: Breast cancer dataset used for experiments
	n. travel_insurance: Travel Insurance dataset used for experiments


	ANN experiments 

	o. breast_cancer_ANN.py: ANN run for breast cancer dataset 
	p. travel_insurance_ANN.py: ANN run for travel insurance dataset 

