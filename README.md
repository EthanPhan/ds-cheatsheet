# Data Science Cheat Sheet

## Model selection:
### Methods:
**Best subset**: $2^p$ models.
* Prone to over-fitting: the search space is huge => find a model that fit the training data by chance without generalization.

**Stepwise selection**: $1 + \frac{p(p+1)}{2}$ models.

Doesn't guarantee the best model because there are correlation between predictors.
* Forward selection: Start with $M_0$
* Backward selection: Start with $M_p$. It require the number of sample is larger than number of variable (se we can fit the full model)

**Shrinkage methods**:

Ridge regression (L2) and Lasso (L1): Impose some constrains on the objective function.

Need standardizing the predictors.

Lasso tend to make the model sparse => better if the true model is sparse.

### Criteria:
Cannot use RSS or $R^2$ since they are related to the training error (calculated using training data). => Always choose the biggest model.

####Adjusted $R^2$:

$Adjusted-R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)}$

where TSS is the total sum of square:

$TSS = \sum_{1}^{n}(y_i - \bar{y}_i)^2$

We want to choose to model with biggest $adjusted-R^2$.

Only works with linear regression.

####Estimated test error
##### Making adjustment to the training error to account for the bias caused by overfitting.


**$C_p$**

$C_p = \frac{1}{n}(RSS + 2d\hat{\sigma}^2)$
* d: # of parameters used.
* $\hat{\sigma}^2$ estimate of variance of error $\epsilon$.

$C_p$ is restricted to the case where n>p so that we can estimate $\hat{\sigma}^2$ (by fitting a full model then use the residuals). If n and p are close, the $\hat{\sigma}^2$ might be too low.

**AIC**

AIC used with likelihood function:

$AIC = - 2log(L) + 2d$
* L is the maximized value of the likelihood function.
* d is the number of parameters.

**BIC**

$BIC = \frac{1}{n}(RSS + log(n)d\hat{\sigma}^2)$
* d: # of parameters used.
* $\hat{\sigma}^2$ estimate of variance of error $\epsilon$.

BIC places a heavier penalties on large models compared to AIC or $C_p$


##### Directly: Use validation set, test set or cross validation.
 **Cross-validation**: k-fold. If k = n we have leave on out cross validation.

 This method doesn't require an estimate of error variance.

------------------------------

## Bagging, Boosting:

### Bagging

Use a group of model to achieve higher accuracy. Each model could have different:
* training sets (using bootstrapping or k fold).
* architectures.
* sets of parameters.

Help reduce variance of the model.


### Boosting

*Not to be mistaken with bootstrapping which is a resampling method along with cross-validation to provide and estimated test error.*

Train a sequence of *simple* models. The new model focus on the residuals or the miss-classified samples. Combine all the model like bagging.

Stop by monitoring performance on holdout set.

AdaBoost.
XGBoost

------------------------------

## Tree-based model

### Decision Trees

**Metric to measure impurity of a node**

Gini method:

$Imp = 1 - \sum_{i=1}^{C}p^2(c_i)$

Where $p(c_i)$ is the probability of class $c_i$ in the node.

**How do we choose the predictor to split?**

At each split we choose the predictor (the variable) that gives the lowest total Gini impurity if split. The total Gini impurity produced by a split is the weighted sum of $Imp$ of each its leaf nodes.

**How do we choose the cutoff (threshold to split)**

* With categorical variable: Compute the impurity for each choice and each combination. Get the lowest impurity.

* With numerical data:
  * Sort the data.
  * Compute mean of adjacent values.
  * Split using each of the above means and compute the Gini impurity
  * Use the value that produce the lowest impurity.

### Random Forest

**Step 1**: Create a bootstrapped dataset.

**Step 2**: Create a decision tree using the bootstrapped dataset. At each split, only consider a **randomly selected subset** of variable as candidates for the splits.

**Step 3**: Repeat step 1 and 2.

To validate: using Out-Of-Bag dataset (since we use bootstrapping at each iteration)

------------------------------

## Logistic regression

Different from linear regression:

| | Logistic Regression | Linear regression |
|---|---------------------|-------------------|
| Purpose | For Qualitative target | For quantitative targe  |
|Fit model| Maximum likelihood | Least square|
|Compare model | Cannot use $R^2$ since there's no RSS  | $R^2$ |

To test the significant of a parameter: Use t-test: Logistic regression can be transform into linear regression.

------------------------------

## ROC

The curve defined by plotting True Positive rate against False Negative rate (1 - recall) at every possible threshold.

Sometimes people use $precision = \frac{True Positive}{True Positive + False Positive}$ instead of  False Negative rate. This is useful when the dataset is imbalanced. It depend of the data to choose which to use.

------------------------------

## Dimension Reduction

### Principal Component Analysis (PCA)

Use for unsupervised learning.

Project the data onto a subspace that retains the highest variance of the data.

* Center the variable around the origin and calculate the covariance matrix of those variable.
* Compute eigenvalues and corresponding eigenvectors of this covariance matrix.
* Normalize each vector to unit vector.
* Pick the the vectors that have the highest eigenvalue

### Singular value decomposition (SVD)

Any matrix A can be factorized as:
$$A = U S V^T$$
where U and V are orthogonal matrices with orthonormal eigenvectors chosen from AAᵀ and AᵀA respectively. S is a diagonal matrix with r elements equal to the root of the positive eigenvalues of AAᵀ or Aᵀ A (both matrics have the same positive eigenvalues anyway)

SVD can be used for dimension reduction.

### Linear Discriminant Analysis (LDA)

For supervised data

------------------------------

## Outlier detection Methods

### Using Quartile range
$$IQR = Q_3 - Q_1$$
where $Q_1$ and $Q_3$ are the median of the first half and second half of the sample divided by the median of all samples.

Outliers are samples that smaller than $Q_1 - 1.5 * IQR$ or bigger than $Q_3 + 1.5 * IQR$.

### Z-score

How many standard deviations a data point is from the sample’s mean, assuming a gaussian distribution?
$$z = \frac{x - \hat{\mu}}{\hat{\sigma}}$$

If z score of a sample exceeds a certain threshold, it's considered an outlier.

Some common thresholds: 2.5, 3, 3.5, or even more.

### Dbscan (Density Based Spatial Clustering of Applications with Noise)

Dbscan is a density based clustering algorithm, it is focused on finding neighbors by density (MinPts) on an ‘n-dimensional sphere’ with radius $\epsilon$. A cluster can be defined as the maximal set of ‘density connected points’ in the feature space.

Dbscan then defines different classes of points:
* **Core point**: A is a core point if its neighborhood (defined by ɛ) contains at least the same number or more points than the parameter MinPts.
* **Border point**: C is a border point that lies in a cluster and its neighborhood does not contain more points than MinPts, but it is still ‘density reachable’ by other points in the cluster.
* **Outlier**: N is an outlier point that lies in no cluster and it is not ‘density reachable’ nor ‘density connected’ to any other point. Thus this point will have “his own cluster”.

A good approach is to try values of $\epsilon$ ranging from 0.25 to 0.75.

### Isolation Forests

Isolation forest’s basic principle is that outliers are few and far from the rest of the observations.

For a tree:
* Get a sample of training dataset
* Randomly select a feature (variable)
* Randomly pick a value of that feature between its min and max and split.
* Repeat for all samples.

Then for prediction, it compares an observation against that splitting value in a “node”, that node will have two node children on which another random comparisons will be made. The number of “splittings” made by the algorithm for an instance is named: “path length”. As expected, outliers will have shorter path lengths than the rest of the observations.

A forest is build by constructing many trees and average them out.

An outlier score can computed for each observation:
$$s(x,n)=2^{\frac{E(h(x))}{c(n)}}$$
where $h(x)$ is the path length of sample $x$ and $c(n)$ is the maximum path length of a binary tree, $n$ is the number of leaf nodes.

---------------
## Anomaly detection

### Simple statistical methods
* Low pass filter: flag the data points that deviate from common statistical properties of a distribution, including mean, median, mode, and quantiles. For time series, we can use rolling window (moving average)
* Kalman filter.

**Challenges:**

The low pass filter allows you to identify anomalies in simple use cases, but there are certain situations where this technique won't work. Here are a few:  

- The data contains noise which might be similar to abnormal behavior, because the boundary between normal and abnormal behavior is often not precise.
- The definition of abnormal or normal may frequently change, as malicious adversaries constantly adapt themselves. Therefore, the threshold based on moving average may not always apply.
- The pattern is based on seasonality. This involves more sophisticated methods, such as decomposing the data into multiple trends in order to identify the change in seasonality.

### Machine learning methods

* k-Nearest Neighbors.
* SVM or other classification methods.
* Isolation Forest.

For financial fraud detection we can use Graph Convolutional Neural Network.

------------------------

## T test

The t test tells you how significant the differences between groups are; In other words it lets you know if those differences (measured in means/averages) could have happened by chance.

**Example**: Student’s T-tests can be used in real life to compare means. For example, a drug company may want to test a new cancer drug to find out if it improves life expectancy. In an experiment, there’s always a control group (a group who are given a placebo, or “sugar pill”). The control group may show an average life expectancy of +5 years, while the group taking the new drug might have a life expectancy of +6 years. It would seem that the drug might work. But it could be due to a fluke. To test this, researchers would use a Student’s t-test to find out if the results are repeatable for an entire population.

### T score

The t score is a ratio between the difference between two groups and the difference within the groups. The larger the t score, the more difference there is between groups. The smaller the t score, the more similarity there is between groups. A t score of 3 means that the groups are three times as different from each other as they are within each other. When you run a t test, the bigger the t-value, the more likely it is that the results are repeatable.

* A large t-score tells you that the groups are different.
* A small t-score tells you that the groups are similar.

### T-Values and P-values

How big is “big enough”? Every t-value has a p-value to go with it. A p-value is the probability that the results from your sample data occurred by chance. P-values are from 0% to 100%. They are usually written as a decimal. For example, a p value of 5% is 0.05. Low p-values are good; They indicate your data did not occur by chance. For example, a p-value of .01 means there is only a 1% probability that the results from an experiment happened by chance. In most cases, a p-value of 0.05 (5%) is accepted to mean the data is valid.

--------------------

## Chi square test

There are two types of chi-square tests. Both use the chi-square statistic and distribution for different purposes:

* A **chi-square goodness of fit** test determines if a sample data matches a population. For more details on this type, see: Goodness of Fit Test.
* A **chi-square test for independence** compares two variables in a contingency table to see if they are related. In a more general sense, it tests to see whether distributions of categorical variables differ from each another.
  * A very small chi square test statistic means that your observed data fits your expected data extremely well. In other words, there is a relationship.
  * A very large chi square test statistic means that the data does not fit very well. In other words, there isn’t a relationship.

The formula for the chi-square statistic used in the chi square test is:
$$\chi_c^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

- $c$ is the degrees of freedom.
- $O$ is your observed value and $E$ is your expected value.

The summation symbol means that you’ll have to perform a calculation for every single data item in your data set.

A chi square test will give you a **p-value**. The p-value will tell you if your test results are significant or not. In order to perform a chi square test and get the p-value, you need two pieces of information:

* **Degrees of freedom**. That’s just the number of categories minus 1.
* The alpha level($\alpha$) or the significant level. The usual alpha level is 0.05 (5%), but you could also have other levels like 0.01 or 0.10.

### Pearson's chi square test (goodness of fit)

Chi-Square goodness of fit test is a non-parametric test that is used to find out how the observed value of a given phenomena is significantly different from the expected value.  In Chi-Square goodness of fit test, the term goodness of fit is used to compare the observed sample distribution with the expected probability distribution.  Chi-Square goodness of fit test determines how well theoretical distribution (such as normal, binomial, or Poisson) fits the empirical distribution. In Chi-Square goodness of fit test, sample data is divided into intervals. Then the numbers of points that fall into the interval are compared, with the expected numbers of points in each interval.

**Step 1**: Set up the hypothesis for Chi-Square goodness of fit test:

* A. **Null hypothesis**: In Chi-Square goodness of fit test, the null hypothesis assumes that there is no significant difference between the observed and the expected value.
* B. **Alternative hypothesis**: In Chi-Square goodness of fit test, the alternative hypothesis assumes that there is a significant difference between the observed and the expected value.

**Step 2**: Compute the value of Chi-Square goodness of fit test using the following formula:
$$\chi^2 = \sum_{i}\left [ \frac{(O_i - E_i)^2}{E_i} \right ]$$
where $\chi^2$ is the goodness of fit, $O_i$ is the observed value at interval $ith$ and $E_i$ is the expected value of interval $ith$.

**Degree of freedom:** $d = n - 1 - k$ where n is the number of intervals (or number of categories of the variable), k is the number of parameters that we calculated using the data. For example, if Null hypothesis is that the data follow normal distribution => we need to calculate mean and variance using the data in order to be able to perform the test => k=2.

### Chi squared test of independence.

The Chi-Square test of independence is used to determine if there is a significant relationship between two nominal (categorical) variables.  The frequency of each category for one nominal variable is compared across the categories of the second nominal variable.  The data can be displayed in a contingency table where each row represents a category for one variable and each column represents a category for the other variable.

For example, say a researcher wants to examine the relationship between gender (male vs. female) and empathy (high vs. low).  The chi-square test of independence can be used to examine this relationship.  The null hypothesis for this test is that there is no relationship between gender and empathy.  The alternative hypothesis is that there is a relationship between gender and empathy (e.g. there are more high-empathy females than high-empathy males).

The chi-squared statistics for this test can be calculated using

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c}\left [ \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \right ]$$

where $O_{ij}$ is the observed value of the two variables; $E_{ij}$ is the expected value of the two variables under the null hypothesis.

Degree of freedom is calculated by using the following formula:
DF = (r-1)(c-1)

-----------------

## Model overfitting

What causes overfitting?

### Solutions:
##### Adding more data/ data augmentation
##### Use other architectures
##### Bagging, bootstrapping, cross-validation (k-fold), boosting
##### Remove features (resize the image ...)
##### Early stopping
##### Dropout
The term “dropout” refers to dropping out units (hidden and visible) in a neural network.

Dropout prevents overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently.

**At test time**: Use a single neural net at test time without dropout. The weights of this network are scaled-down versions of the trained weights. If a unit is retained with probability p during training, the outgoing weights of that unit are multiplied by p at test time. This ensures that for any hidden unit the expected output (under the distribution used to drop units at training time) is the same as the actual output at test time.

##### L1 norm
Puts a constrain on the weights of the model. L1 norm will push a lot of parameters at exact 0. This is more suitable compared to L2 if the true model is sparse.
##### L2 norm
Like L1 norm, L2 norm puts a constrain on the weights of the model. L2 norm push parameters toward zero but not exactly at 0 (derivative of L2 is very small around 0)
##### Adding noise
* Add noise to input
* Add noise to the label
* Add noise to the gradient

---------------------------

**Why do we need to shuffle data when training using mini batch?**

During training, we try to minimize the loss function
$$J(\theta) = \frac{1}{N}\sum_{i=1}^{N}L(\theta, x_i)$$
where $N$ is the number of samples in the training set and $x_i$ is the $ith$ training sample.

When we use mini-batch for training, what we actually minimizes is a series of approximated function
$$\hat{J}_1(\theta) = \frac{1}{M}\sum_{i=1}^{M}L(\theta, x_i)$$
$$\hat{J}_2(\theta) = \frac{1}{M}\sum_{i=M+1}^{2M}L(\theta, x_i)$$
$$\hat{J}_3(\theta) = \frac{1}{M}\sum_{i=2M+1}^{3M}L(\theta, x_i)$$
$$...$$

where $M$ is batch size. Note that those are functions of $\theta$ - the parameters of the models, not functions of $x_i$. Each of those functions is an **estimate** of $J(\theta)$

If the training set is not shuffled after each epoch which means for every epoch, $\hat{J}_1(\theta)$ get the same parameters (same set of $x_i$) => They are **fixed functions** (not sure i'm using the right word here). We can prove that, in this case, they are all **biased estimate** of the true $J(\theta)$.

If the training set is shuffled, each epoch those  $\hat{J}$ function get a different set of parameters $x_i$. It's easy to show that this time:
$$E_{x_i \sim training \ set}[\hat{J}_j(\theta)] = J(\theta)$$

This lead to better update.

-------------------------------

## Data normalization

### Data standardization
The Features will be rescaled so that they’ll have the properties of a standard normal distribution with $\mu = 0$ and $\sigma = 1$. We can achieve it by using the following equation
 $$z_i = \frac{x_i - \bar{x}}{s}$$

### Min-max scaling (normalization)
This approach scales the features to a fix range, usually [0, 1] or [-1, 1].
The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard deviations, which can suppress the effect of outliers.

### Effects
* Data normalization makes the the loss functions 'smoother' in the hyper space => gradient descent converges much faster, and the training process is much more statble
* The majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

-----------------------------
## Batch normalization

This has the same effects as data normalization, but it is applied to the intermediate output inside the model instead of the input. This is because after one or more layer, the data starts to suffer from what author called 'covariance shift' which is different features has different scale => take longer to train using gradient descent.

Batch normalization also help avoid vanishing gradient since it prevents the activation from saturating by bounding their inputs.

----------------------------
## Vanishing gradient and Exploding gradient
This is because of the activation functions. For example, sigmoid function saturates out at high (> 6) or low (, -6) input. It's derivative at this region is very low.

We can use ReLu to prevent vanishing gradient and exploding gradient.
