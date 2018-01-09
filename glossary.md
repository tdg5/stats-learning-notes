---
layout: default
use_math: true
---

# Glossary

<a id="backwards-selection"></a>
**[Backwards Selection][#backwards-selection]**: A variable selection method
that begins with a model that includes all the predictors and proceeds by
removing the variable with the highest [p-value][#p-value] each iteration until
some stopping condition is met. Backwards selection cannot be used when $$ p > n
. $$

<a id="bayes-classifier"></a>
**[Bayes Classifier][#bayes-classifier]**: A very simple classifier that assigns
each observation to the most likely class given its predictor variables.

In Bayesian terms, an observation should be classified for the predictor vector
$$ x_{0} $$ to the class $$ j $$ for which

$$ \normalsize \mathrm{Pr}(Y=j|X=x_{0}) $$

is largest. That is, the class for which the conditional probability that $$ Y=j
$$, given the observed predictor vector $$ x_{0} , $$ is largest.

<a id="bayes-decision-boundary"></a>
**[Bayes Decision Boundary][#bayes-decision-boundary]**: The threshold inherent
to a two-class [Bayes classifier][#bayes-classifier] where the classification
probability is exactly 50%.

<a id="bayes-error-rate"></a>
**[Bayes Error Rate][#bayes-error-rate]**: The [Bayes
classifier][#bayes-classifier] yields the lowest possible test error rate since
it will always choose the class with the highest probability. The Bayes error
rate can be stated formally as

$$ \normalsize 1 - \mathrm{E} \lgroup \max_{j} \mathrm{Pr}(Y=j|X) \rgroup . $$

The Bayes error rate can also be described as the ratio of observations that lie
on the "wrong" side of the decision boundary.

<a id="bayes-theorem"></a>
**[Bayes Theorem][#bayes-theorem]**: Describes the probability of an event,
based on prior knowledge of conditions that might be related to the event. Also
known as Bayes' law or Bayes' rule.

Bayes' theorem is stated mathematically as

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

where $$ A $$ and $$ B $$ are events and $$ P(B) $$ is greater than zero.

<a id="bias"></a>
**[Bias][#bias]**: The error that is introduced by approximating a potentially
complex function using a simple model. More flexible models tend to have less
bias.

<a id="bias-variance-trade-off"></a>
**[Bias-Variance Trade-Off][#bias-variance-trade-off]**: The relationship
between [bias][#bias], [variance][#variance], and [test mean squared
error][#test-mean-squared-error], called a trade-off because it is a challenge
to find a model that has a low test mean squared error and both a low variance
and a low squared bias.

<a id="bootstrap"></a>
**[Bootstrap][#bootstrap]**: A widely applicable resampling method that can be
used to quantify the uncertainty associated with a given estimator or
statistical learning approach, including those for which it is difficult to
obtain a measure of variability.

The bootstrap generates distinct data sets by repeatedly sampling observations
from the original data set. These generated data sets can be used to estimate
variability in lieu of sampling independent data sets from the full population.

The sampling employed by the bootstrap involves randomly selecting $$ n $$
observations with replacement, which means some observations can be selected
multiple times while other observations are not included at all.

This process is repeated $$ B $$ times to yield $$ B $$ bootstrap data sets, $$
Z^{*1}, Z^{*2}, ..., Z_{*B} , $$ which can be used to estimate other quantities
such as standard error.

For example, the estimated standard error of an estimated quantity $$
\hat{\alpha} $$ can be computed using the bootstrap as follows:

$$ SE_{B}(\hat{\alpha}) = \sqrt{\frac{1}{B-1}\sum_{r=1}^{B}(\hat{\alpha}^{*r} -
\frac{1}{B}\sum_{s=1}^{B}\hat{\alpha}^{*s})^{2}} $$

<a id="classification-problem"></a>
**[Classification Problem][#classification-problem]**: A class of problem that
is well suited to statistical techniques for determining if an observation is a
member of a particular class or which of a number of classes the observation
belongs to.

<a id="cluster-analysis"></a>
**[Cluster Analysis][#cluster-analysis]**: The task of grouping a set of
observations in such a way that the observations in the same group or cluster
are more similar, in some sense, to each other than those observations in other
groups or clusters.

<a id="coefficient"></a>
**[Coefficient][#coefficient]**: A number or symbol representing a number that
is  multiplied with a variable or an unknown quantity in an algebraic term.

<a id="collinearity"></a>
**[Collinearity][#collinearity]**: The situation in which two or
more predictor variables are closely related to one another.

Collinearity can pose problems for linear regression because it can make it hard
to determine the individual impact of collinear predictors on the response.

Collinearity reduces the accuracy of the regression coefficient estimates, which
in turn causes the standard error of $$ \beta_{j} $$ to grow. Since the
T-statistic for each predictor is calculated by dividing $$ \beta_{j} $$ by its
standard error, collinearity results in a decline in the true T-statistic. This
may cause it to appear that $$ \beta_{j} $$ and $$ x_{j} $$ are related to the
response when they are not. As such, collinearity reduces the effectiveness of
the null hypothesis. Because of all this, it is important to address possible
collinearity problems when fitting the model.

One way to detect collinearity is to generate a correlation matrix of the
predictors. Any element in the matrix with a large absolute value indicates
highly correlated predictors. This is not always sufficient though, as it is
possible for collinearity to exist between three or more variables even if no
pair of variables have high correlation. This scenario is known as
[multicollinearity][#multicollinearity].

<a id="confidence-interval"></a>
**[Confidence Interval][#confidence-interval]**: A range of values such that
there's an X% likelihood that the range will contain the true, unknown value of
the [parameter][#parameter]. For example, a 95% confidence interval is a range
of values such that there's a 95% chance that the range contains the true
unknown value of the parameter.

<a id="confounding"></a>
**[Confounding][#confounding]**: In general, the scenario in which the result
obtained with a single predictor does not match the result with multiple
predictors, especially when there is correlation among the predictors. More
specifically, confounding describes situations in which the experimental
controls do not adequately allow for ruling out alternative explanations for the
observed relationship between the predictors and the response.

<a id="correlation"></a>
**[Correlation][#correlation]**: A measure of the linear relationship between $$
X $$ and $$ Y , $$ calculated as

$$ \normalsize \mathrm{Cor}(X,Y) = \frac{\sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} -
\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_{i} -
\bar{x})^{2}}\sqrt{\sum_{i=1}^{n}(y_{i}-\bar{y})^{2}}} . $$

<a id="cross-validation"></a>
**[Cross Validation][#cross-validation]**: A resampling method that can be
used to estimate a given statistical methods test error or to determine the
appropriate amount of flexibility.

Cross validation can be used both to estimate how well a given statistical
learning procedure might perform on new data and to estimate the minimum point
in the estimated test mean squared error curve, which can be useful when
comparing statistical learning methods or when comparing different levels of
flexibility for a single statistical learning method.

Cross validation can also be useful when $$ Y $$ is qualitative, in which case
the number of misclassified observations is used instead of mean squared error.

<a id="curse-of-dimensionality"></a>
**[Curse of Dimensionality][#curse-of-dimensionality]**: Refers to various
phenomena that arise when analyzing and organizing data in high-dimensional
settings that do not occur in low-dimensional settings. The common theme of
these problems is that when the dimensionality increases, the volume of the
space increases so fast that the available data become sparse which can be
problematic for any method that requires statistical significance because the
amount of data needed to support a statistically sound and reliable result often
grows exponentially with the dimensionality.

<a id="degrees-of-freedom"></a>
**[Degrees of Freedom][#degrees-of-freedom]**: A numeric value that quantifies
the number of values in the model that are free to vary. The degrees of freedom
is a quality that summarizes the flexibility of a curve.

<a id="density-function"></a>
**[Density Function][#density-function]**: A function whose value for any given
sample in the sample space (the set of possible values taken by the random
variable) can be interpreted as providing a relative likelihood that the value
of the random variable would equal that sample.

The density function of $$ X $$ for an observation that comes from the kth class
is defined as

$$ \normalsize f_{k}(X) = \mathrm{Pr}(X=x|Y=k) . $$

This means that $$ f_{k}(X) $$ should be relatively large if there's a high
probability that an observation from the kth class features $$ X = x . $$
Conversely, $$ f_{k}(X) $$ will be relatively small if it is unlikely that an
observation in class k would feature $$ X = x . $$

<a id="discriminant-analysis"></a>
**[Discriminant Analysis][#discriminant-analysis]**: An alternative to
regression analysis applied to discrete dependent variables and concerned with
separating sets of observed values into classes.

<a id="dummy-variable"></a>
**[Dummy Variable][#dummy-variable]**: A derived variable taking on a value of 0
or 1 to indicate membership in some mutually exclusive category or class.
Multiple dummy variables can be used in conjunction to model classes with more
than two possible values. Similar to and often used interchangeably with the
term [indicator variable][#indicator-variable]. Dummy variables make it easy to
mix quantitative and qualitative predictors.

An example dummy variable encoding:

$$ \normalsize X_{i} = \left\{ \begin{array}{cc}
  0&\mathrm{if\ p_{i}\ =\ class\ A}\\
  1&\mathrm{if\ p_{i}\ =\ class\ B}
\end{array} \right. $$

When dummy variables are used to model classes with more than two possible
values, the number of dummy variables required will always be one less than the
number of values that the predictor can take on.

For example, with a predictor that can take on three values, the following
coding could be used:

$$ \normalsize X_{i1} = \left\{ \begin{array}{cc}
  1&\mathrm{if\ p_{i}\ =\ class\ A}\\
  0&\mathrm{if\ p_{i}\ \ne\ class\ A}
\end{array} \right. $$

$$ \normalsize X_{i2} = \left\{ \begin{array}{cc}
  1&\mathrm{if\ p_{i}\ =\ class\ B}\\
  0&\mathrm{if\ p_{i}\ \ne\ class\ B}
\end{array} \right. $$

<a id="error-term"></a>
**[Error Term][#error-term]**:

<a id="f-distribution"></a>
**[F-Distribution][#f-distribution]**: A continuous probability distribution
that arises frequently as the null distribution of a test statistic, most
notably in the analysis of variance.

<a id="f-statistic"></a>
**[F-Statistic][#f-statistic]**: A test statistic which adheres to an
[F-distribution][#f-distribution] under the null hypothesis which is useful to
assess model fit.

The F-statistic can be computed as

$$ \normalsize \mathrm{F} = \frac{(\mathrm{TSS} -
\mathrm{RSS})/p}{\mathrm{RSS}/(n - p - 1)} = \frac{\frac{\mathrm{TSS} -
\mathrm{RSS}}{p}}{\frac{\mathrm{RSS}}{n - p
- 1}} $$

where, again,

$$ \normalsize \mathrm{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y}_{i})^{2} $$

and

$$ \normalsize \mathrm{RSS} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^2 $$

If the assumptions of the linear model, represented by the alternative
hypothesis, are true it can be shown that

$$ \normalsize \mathrm{E}\{\frac{\mathrm{RSS}}{n - p - 1}\} = \sigma^{2} $$

Conversely, if the null hypothesis is true, it can be shown that

$$ \normalsize \mathrm{E}\{\frac{\mathrm{TSS} - \mathrm{RSS}}{p}\} = \sigma^{2} $$

This means that when there is no relationship between the response and the
predictors the F-statisitic takes on a value close to $$ 1 . $$

Conversely, if the alternative hypothesis is true, then the F-statistic will
take on a value greater than $$ 1 . $$

When $$ n $$ is large, an F-statistic only slightly greater than $$ 1 $$ may
provide evidence against the null hypothesis. If $$ n $$ is small, a large
F-statistic is needed to reject the null hypothesis.

<a id="forward-selection"></a>
**[Forward Selection][#forward-selection]**: A variable selection method that
begins with a [null model][#null-model], a model that has an intercept
but no predictors, and attempts $$ p $$ simple linear regressions, keeping
whichever predictor results in the lowest [residual sum of
squares][#residual-sum-of-squares]. In this fashion, the predictor yielding the
lowest residual sum of squares is added to the model one-by-one until some
halting condition is met. Forward selection is a greedy process that may include
extraneous variables.

<a id="gaussian-distribution"></a>
**[Gaussian Distribution][#gaussian-distribution]**: A theoretical frequency
distribution represented by a normal curve or bell curve. Also known as a
normal distribution.

<a id="heteroscedasticity"></a>
**[Heteroscedasticity][#heteroscedasticity]**: A characteristic of a collection
of random variables in which there are sub-populations that have different
variability from other sub-populations. Heteroscedasticity can lead to
regression models that seem stronger than they really are since standard errors,
confidence intervals, and hypothesis testing all assume that error terms have a
constant variance.

<a id="hierarchical-principle"></a>
**[Hierarchical Principle][#hierarchical-principle]**: A guiding philosophy that
states that, when an interaction term is included in the model, the main effects
should also be included, even if the p-values associated with their coefficients
are not significant. The reason for this is that $$ X_{1}X_{2} $$ is often
correlated with $$ X_{1} $$ and $$ X_{2} $$ and removing them tends to change
the meaning of the interaction.

If $$ X_{1}X_{2} $$ is related to the response, then whether or not the
coefficient estimates of $$ X_{1} $$ or $$ X_{2} $$ are exactly zero is of
limited interest.

<a id="high-leverage"></a>
**[High Leverage][#high-leverage]**: In contrast to outliers which relate to
observations for which the response $$ y_{i} $$ is unusual given the predictor
$$ x_{i} $$, observations with high leverage are those that have an unusual
value for the predictor $$ x_{i} $$ for the given response $$ y_{i} . $$

High leverage observations tend to have a sizable impact on the estimated
regression line and as a result, removing them can yield improvements in model
fit.

For simple linear regression, high leverage observations can be identified as
those for which the predictor value is outside the normal range. With multiple
regression, it is possible to have an observation for which each individual
predictor's values are well within the expected range, but that is unusual in
terms of the combination of the full set of predictors.

To qualify an observation's leverage, the leverage statistic can be computed.

A large leverage statistic indicates an observation with high leverage.

For simple linear regression, the leverage statistic can be computed as

$$ \normalsize h_{i} = \frac{1}{n} + \frac{(x_{i} -
\bar{x})^{2}}{\sum_{j=1}^{n}(x_{j} - \bar{x})^{2}} . $$

The leverage statistic always falls between $$ \frac{1}{n} $$ and $$ 1 $$ and
the average leverage is always equal to $$ \frac{p + 1}{n} . $$ So, if an
observation has a leverage statistic greatly exceeds $$ \frac{p + 1}{n} $$ then
it may be evidence that the corresponding point has high leverage.

<a id="hypothesis-testing"></a>
**[Hypothesis Testing][#hypothesis-testing]**: The process of applying the
scientific method to produce, test, and iterate on theories. Typical steps
include: making an initial assumption; collecting evidence (data); and deciding
whether to accept or reject the initial assumption, based on the available
evidence (data).

<a id="input"></a>
**[Input][#input]**: The variables that contribute to the value of the response
or dependent variable in a given function. Also known as predictors, independent
variables, or features. Input variables may be qualitative or quantitative.

<a id="indicator-variable"></a>
**[Indicator Variable][#indicator-variable]**: A derived variable taking on a
mutually exclusive value of 0 or 1 based on an associated indicator function
which typically returns 0 to indicate the absence of some property and 1 to
indicate the presence of that property. Similar to and often used
interchangeably with the term [dummy variable][#dummy-variable], especially when
dealing with classes with only two possible values. Indicator variables make it
easy to mix qualitative and quantitative predictors.

An example indicator variable encoding:

$$ \normalsize X_{i} = \left\{ \begin{array}{cc}
  1&\mathrm{if\ p_{i}\ =\ class\ A}\\
  0&\mathrm{otherwise}
\end{array} \right. $$

<a id="interaction-term"></a>
**[Interaction Term][#interaction-term]**: A derived predictor that gets its
value from computing the product of two associated predictors with the goal of
better capturing the interaction between the two given predictors.

A simple linear regression model account for interaction between the predictors
would look like

$$ \normalsize \mathrm{Y} = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} +
\beta_{3}X_{1}X_{2} + \epsilon $$

where $$ \beta_{3} $$ can be interpreted as the increase in effectiveness of $$
\beta_{1} $$ given a one-unit increase in $$ \beta_{2} $$ and vice-versa.

It is sometimes possible for an interaction term to have a very small p-value
while the associated main effects, $$ X_{1}, X_{2}, etc. $$, do not. Even in
such a scenario the main effects should still be included in the model due to
the [hierarchical principle][#hierarchical-principle].

<a id="intercept"></a>
**[Intercept][#intercept]**: In a linear model, the value of the dependent
variable, $$ Y , $$ when the independent variable, $$ X , $$ is equal to zero.
Also described as the point at which a given line intersects with the x-axis.

<a id="irreducible-error"></a>
**[Irreducible Error][#irreducible-error]**: A random error term that is
independent of the independent variables with a mean roughly equal to zero
that is intended to account for inaccuracies in a function which may arise from
unmeasured variables or unmeasured variation. The irreducible error will always
enforce an upper bound on the accuracy of attempts to predict the dependent
variable.

<a id="k-fold-cross-validation"></a>
**[K-Fold Cross Validation][#k-fold-cross-validation]**: A resampling method
that operates by randomly dividing the set of observations into $$ K $$ groups
or folds of roughly equal size. Similar to [leave-one-out cross
validation][#leave-one-out-cross-validation], each of the $$ K $$ folds is used
as the [validation set][#validation-set] while the other $$ K - 1 $$ folds are
used as the test set to generate $$ K $$ estimates of the test error.  The
K-fold cross validation estimated test error comes from the average of these
estimates.

$$ CV(k) = \frac{1}{k}\sum_{i=1}^{k} MSE_{i} $$

It can be shown that leave-one-out cross validation is a special case of K-fold
cross validation where $$ K = n . $$

Typical values for $$ K $$ are 5 or 10 since these values require less
computation than when $$ K $$ is equal to $$ n . $$

There is a [bias-variance trade-off][#bias-variance-trade-off] inherent to the
choice of $$ K $$ in K-fold cross validation. Typically, values of $$ K = 5 $$
or $$ K = 10 $$ are used as these values have been empirically shown to produce
test error rate estimates that suffer from neither excessively high bias nor
very high variance.

In terms of bias, leave-one-out cross validation is preferable to K-fold cross
validation and K-fold cross validation is preferable to the validation set
approach.

In terms of variance, K-fold cross validation where $$ K < n $$ is preferable to
leave-one-out cross validation and leave-one-out cross validation is preferable
to the validation set approach.


<a id="k-nearest-neighbors-classifier"></a>
**[K-Nearest Neighbors Classifier][#k-nearest-neighbors-classifier]**: A
classifier that takes a positive integer $$ K $$ and first identifies the $$ K
$$ points that are nearest to $$ x_{0} , $$ represented by $$ N_{0} . $$ It next
estimates the conditional probability for class $$ j $$ based on the fraction of
points in $$ N_{0} $$ that have a response equal to $$ j . $$ Formally, the
estimated conditional probability can be stated as

$$ \normalsize \mathrm{Pr}(Y=j|X=x_{0}) =
\frac{1}{k} \sum_{i \in N_{0}}\mathrm{I}(y_{i}=j) $$

The K-Nearest Neighbor classifier then applies [Bayes theorem][#bayes-theorem]
and yields the classification with the highest probability.

Despite its simplicity, the K-Nearest Neighbor classifier often yields results
that are surprisingly close to the optimal Bayes classifier.

The choice of $$ K $$ can have a drastic effect on the yielded classifier, as $$
K $$ controls the bias-variance trade-off for the classifier.

<a id="k-nearest-neighbors-regression"></a>
**[K-Nearest Neighbors Regression][#k-nearest-neighbors-regression]**: A
non-parametric method akin to linear regression.

Given a value for $$ K $$ and a prediction point $$ x_{0} $$, k-nearest
neighbors regression first identifies the $$ K $$ observations that are closest
to $$ x_{0} $$, represented by $$ N_{0} . $$ $$ f(x_{0}) $$ is then estimated
using the average of $$ N_{0i} $$ like so

$$ \normalsize \hat{f}(x_{0}) = \frac{1}{k}\sum_{x_{i} \in N_{0}}y_{i} . $$

In higher dimensions, K-nearest neighbors regression often performs worse than
linear regression. This is often due to combining too small an $$ n $$ with too
large a $$ p $$, resulting in a given observation having no nearby neighbors.
This is often called the [curse of dimensionality][#curse-of-dimensionality].

<a id="least-squares-line"></a>
**[Least Squares Line][#least-squares-line]**: The line yielded by least squares
regression,

$$ \normalsize \hat{y_{i}} = \hat{\beta_{0}} + \hat{\beta_{1}}x_{i} , $$

where the coefficients $$ \hat{\beta_{0}} $$ and $$ \hat{\beta_{1}} $$ are
approximations of the unknown coefficients of the [population regression
line][#population-regression-line], $$ \beta_{0} $$ and $$ \beta_{1} . $$

<a id="leave-one-out-cross-validation"></a>
**[Leave-One-Out Cross Validation][#leave-one-out-cross-validation]**: A
resampling method similar to the [validation set][#validation-set] approach,
except instead of splitting the observations evenly, leave-one-out
cross-validation withholds only a single observation for the validation set.
This process can be repeated $$ n $$ times with each observation being withheld
once. This yields $$ n $$ mean squared errors which can be averaged together to
yield the leave-one-out cross-validation estimate of the test mean squared
error.

$$ CV(n) = \frac{1}{n}\sum_{i=1}^{n} MSE_{i} $$

Leave-one-out cross validation has much less bias than the validation set
approach. Leave-one-out cross validation also tends not to overestimate the test
mean squared error since many more observations are used for training. In
addition, leave-one-out cross validation is much less variable, in fact, it
always yields the same result since there's no randomness in the set splits.

Leave-one-out cross validation can be expensive to implement since the model has
to be fit $$ n $$ times. This can be especially expensive in situations where $$
n $$ is very large and/or when each individual model is slow to fit.

In the classification setting, the leave-one-out cross validation error rate
takes the form

$$ CV(n) = \frac{1}{n}\sum_{i=1}^{n}Err_{i} $$

where $$ Err_{i} = I(y \neq \hat{y}_{i}) . $$ The K-fold cross validation error
rate and the validation set error rate are defined similarly.

<a id="likelihood-function"></a>
**[Likelihood Function][#likelihood-function]**: A function often used to
estimate parameters from a set of independent observations. For logistic
regression a common likelihood function takes the form

$$ \normalsize \ell(\beta_{0}, \beta_{1}) = \displaystyle
\prod_{i:y_{i}=1}p(X_{i}) \times \displaystyle \prod_{j:y_{j}=0}(1-p(X_{j})) .
$$

Estimates for $$ \beta_{0} $$ and $$ \beta_{1} $$ are chosen so as to maximize
this likelihood function for logistic regression.

<a id="linear-discriminant-analysis"></a>
**[Linear Discriminant Analysis][#linear-discriminant-analysis]**: While
[logistic regression][#logistic-regression] models the conditional distribution of the response $$ Y $$
given the predictor(s) $$ X , $$ linear discriminant analysis takes the approach
of modeling the distribution of the predictor(s) $$ X $$ separately in each of
the response classes , $$ Y $$, and then uses [Bayes'
theorem][#bayes-theorem] to invert these probabilities to estimate the
conditional distribution.

Linear discriminant analysis is popular when there are more than two response
classes. Beyond its popularity, linear discriminant analysis also benefits from
not being susceptible to some of the problems that logistic regression suffers
from:

- The parameter estimates for logistic regression can be surprisingly unstable
  when the response classes are well separated. Linear discriminant analysis
  does not suffer from this problem.
- Logistic regression is more unstable than linear discriminant analysis when $$
  n $$ is small and the distribution of the predictors $$ X $$ is approximately
  normal in each of the response classes.

<a id="log-odds"></a>
**[Log-Odds][#log-odds]**: Taking a logarithm of both sides of the [logistic
odds][#odds] equation yields an equation for the log-odds or [logit][#logit],

$$ \normalsize \mathrm{log} \bigg \lgroup \frac{p(X)}{1 - p(X)} \bigg \rgroup =
\beta_{0} + \beta_{1}X $$

Logistic regression has log-odds that are linear in terms of $$ X . $$

The log-odds equation for multiple logistic regression can be expressed as

$$ \normalsize p(X) = \frac{e^{\beta_{0} + \beta_{1}X_{1} + \ldots +
\beta_{p}X_{p}}}{1 + e^{\beta_{0} + \beta_{1}X_{1} + \ldots + \beta_{p}X_{p}}} .
$$

<a id="logistic-function"></a>
**[Logistic Function][#logistic-function]**: A function with a common "S" shaped
Sigmoid curve guaranteed to return a value between $$ 0 $$ and $$ 1 $$. For
[logistic regression][#logistic-regression], the logistic function takes the
form

$$ \normalsize p(X) = \frac{e^{\beta_{0} + \beta_{1}X}}{1 + e^{\beta_{0} +
\beta_{1}X}} . $$

<a id="logistic-regression"></a>
**[Logistic Regression][#logistic-regression]**: A regression model where the
dependent variable is categorical or qualitative. Logistic regression models the
probability that $$ y $$ belongs to a particular category rather than modeling
the response itself. Logistic regression uses a [logistic
function][#logistic-function] to ensure a prediction between $$ 0 $$ and $$ 1 .
$$

<a id="logit"></a>
**[Logit][#logit]**: Taking a logarithm of both sides of the [logistic
odds][#odds] equation yields an equation for the [log-odds][#log-odds] or logit.

$$ \normalsize \mathrm{log} \bigg \lgroup \frac{p(X)}{1 - p(X)} \bigg \rgroup =
\beta_{0} + \beta_{1}X $$

Logistic regression has log-odds that are linear in terms of $$ X . $$

<a id="maximum-likelihood"></a>
**[Maximum Likelihood][#maximum-likelihood]**: A strategy utilized by [logistic
regression][#logistic-regression] to estimate regression coefficients.

Maximum likelihood plays out like so: determine estimates for $$ \beta_{0} $$
and $$ \beta_{1} $$ such that the predicted probability of $$ \hat{p}(x_{i}) $$
corresponds with the observed classes as closely as possible. Formally, this
yield an equation called a [likelihood function][#likelihood-function]:

$$ \normalsize \ell(\beta_{0}, \beta_{1}) = \displaystyle
\prod_{i:y_{i}=1}p(X_{i}) \times \displaystyle \prod_{j:y_{j}=0}(1-p(X_{j})) .
$$

Estimates for $$ \beta_{0} $$ and $$ \beta_{1} $$ are chosen so as to maximize
this likelihood function.

Linear regression's least squares approach is actually a special case of maximum
likelihood.

Maximum likelihood is also used to estimate $$ \beta_{0}, \beta_{1}, \ldots,
\beta_{p} $$ in the case of multiple logistic regression.

<a id="mean-squared-error"></a>
**[Mean Squared Error][#mean-squared-error]**: A statistical method for
assessing how well the responses predicted by a modeled function correspond to
the actual observed responses. Mean squared error is calculated by calculating
the average squared difference between the predicted responses and their
relative observed responses, formally,

$$ \normalsize \frac{1}{n} \sum_{i=1}^{n} \big \lgroup y_i - \hat{f}(x_{i}) \big
\rgroup ^{2} . $$

Mean squared error will be small when the predicted responses are close to the
true responses and large if there's a substantial difference between the
predicted response and the observed response for some observations.

<a id="mixed-selection"></a>
**[Mixed Selection][#mixed-selection]**: A variable selection method that begins
with a null model, like [forward selection][#forward-selection], repeatedly
adding whichever predictor yields the best fit. As more predictors are added,
the p-values become larger. When this happens, if the p-value for one of the
variables exceeds a certain threshold, that variable is removed from the model.
The selection process continues in this forward and backward manner until all
the variables in the model have sufficiently low p-values and all the predictors
excluded from the model would result in a high p-value if added to the model.

<a id="model-assessment"></a>
**[Model Assessment][#model-assessment]**: The process of evaluating the
performance of a given model.

<a id="model-selection"></a>
**[Model Selection][#model-selection]**: The process of selecting the
appropriate level of flexibility for a given model.

<a id="multicollinearity"></a>
**[Multicollinearity][#multicollinearity]**: The situation in which collinearity
exists between three or more variables even if no pair of variables have high
correlation.

Multicollinearity can be detected by computing the [variance inflation
factor][variance-inflation-factor].

<a id="multiple-linear-regression"></a>
**[Multiple Linear Regression][#multiple-linear-regression]**: An extension of
simple linear regression that accommodates multiple predictors.

The multiple linear regression model takes the form of

$$ \normalsize Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \ldots +
\beta_{p}X_{p} + \epsilon . $$

$$ X_{j} $$ represents the $$ j \text{th} $$ predictor and $$ \beta_{j} $$
represents the average effect of a one-unit increase in $$ X_{j} $$ on $$ Y $$,
holding all other predictors fixed.

<a id="multiple-logistic-regression"></a>
**[Multiple Logistic Regression][#multiple-logistic-regression]**: An extension
of simple logistic regression that accommodates multiple predictors. Multiple
logistic regression can be generalized as

$$ \normalsize log \bigg \lgroup \frac{p(X)}{1 - p(X)} \bigg \rgroup = \beta_{0}
+ \beta_{1}X_{1} + \ldots + \beta_{p}X_{p} $$

where $$ X = (X_{1}, X_{2}, \ldots, X_{p}) $$ are $$ p $$ predictors.

<a id="multivariate-gaussian-distribution"></a>
**[Multivariate Gaussian Distribution][#multivariate-gaussian-distribution]**: A
generalization of the one-dimensional [Gaussian
distribution][#gaussian-distribution] that assumes that each predictor follows a
one-dimensional normal distribution with some correlation between the
predictors. The more correlation between predictors, the more the bell shape of
the normal distribution will be distorted.

A p-dimensional variable X can be indicated to have a multivariate Gaussian
distribution with the notation $$ X \sim N(\mu, \Sigma) $$ where $$ E(x) = \mu $$
is the mean of $$ X $$ (a vector with p components) and $$ \mathrm{Cov}(X) =
\Sigma $$ is the p x p covariance matrix of $$ X $$.

Multivariate Gaussian density is formally defined as

$$ \normalsize f(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp \big \lgroup
-\frac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x - \mu) \big \rgroup . $$

<a id="non-parametric"></a>
**[Non-Parametric][#non-parametric]**: Not involving any assumptions about the
form or parameters of a function being modeled.

<a id="non-parametric-methods"></a>
**[Non-Parametric Methods][#non-parametric-methods]**: A class of techniques
that don’t make explicit assumptions about the shape of the function being
modeled and instead seek to estimate the function by getting as close to the
training data points as possible without being too coarse or granular,
preferring smoothness instead. Non-parametric methods can fit a wider range of
possible functions since essentially no assumptions are made about the form of
the function being modeled.

<a id="normal-distribution"></a>
**[Normal Distribution][#normal-distribution]**: A theoretical frequency
distribution represented by a normal curve or bell curve. Also known as a
Gaussian distribution.

<a id="null-hypothesis"></a>
**[Null Hypothesis][#null-hypothesis]**:
The most common hypothesis test involves testing the null hypothesis that states

$$ H_{0} $$: There is no relationship between $$ X $$ and $$ Y $$

versus the alternative hypothesis

$$ H_{1} $$: Thee is some relationship between $$ X $$ and $$ Y . $$

In mathematical terms, the null hypothesis corresponds to testing if $$
\beta_{1} = 0 $$, which reduces to

$$ \normalsize Y = \beta_{0} + \epsilon $$

which evidences that $$ X $$ is not related to $$ Y . $$

To test the null hypothesis, it is necessary to determine whether the estimate
of $$ \beta_{1} $$, $$ \hat{\beta_{1}} $$, is sufficiently far from zero to provide
confidence that $$ \beta_{1} $$ is non-zero.

<a id="null-model"></a>
**[Null Model][#null-model]**: In linear regression, a model that includes an
intercept, but no predictors.

<a id="odds"></a>
**[Odds][#odds]**: The [logistic function][#logistic-function] can be rebalanced
to yield

$$ \normalsize \frac{p(X)}{1 - p(X)} = e^{\beta_{0} + \beta_{1}X} . $$

$$ \frac{p(X)}{1 - p(X)} $$ is known as the odds and takes on a value between
$$ 0 $$ and infinity.

As an example, a probability of 1 in 5 yields odds of $$ \frac{1}{4} $$ since
$$ \frac {0.2}{1 - 0.2} = \frac{1}{4} . $$

<a id="outlier"></a>
**[Outlier][#outlier]**: A point for which $$ y_{i} $$ is far from the value
predicted by the model.

Excluding outliers can result in improved residual standard error and improved
$$ \mathrm{R}^{2} $$ values, usually with negligible impact to the least squares
fit.

Residual plots can help identify outliers, though it can be difficult to know
how big a residual needs to be before considering a point an outlier. To address
this, it can be useful to plot the [studentized
residuals][#studentized-residual] instead of the normal residuals. Observations
whose studentized residual is greater than $$ |3| $$ are possible outliers.

Outliers should only be removed when confident that the outliers are due to a
recording or data collection error since outliers may otherwise indicate a
missing predictor or other deficiency in the model.

<a id="output"></a>
**[Output][#output]**: The result of computing a given function with all of the
independent variables replaced with concrete values. Also known as the response
or dependent variable. Output may be qualitative or quantitative.

<a id="overfitting"></a>
**[Overfitting][#overfitting]**: A phenomenon where a model closely matches the
training data such that it captures too much of the noise or error in the data.
This results in a model that fits the training data very well, but doesn't make
good predictions under test or in general. Overfitting refers specifically to
scenarios in which a less flexible model would have yielded a smaller [test mean
squared error][#test-mean-squared-error].

<a id="p-value"></a>
**[P-Value][#p-value]**: When the null hypothesis is true, the probability for a
given model that a statistical summary (such as a [t-statistic][#t-statistic])
would be of equal or of greater magnitude compared to the actual observed
results. Can indicate an association between the predictor and the response if
the value is sufficiently small, further indicating that the null hypothesis may
not be true.

<a id="parameter"></a>
**[Parameter][#parameter]**: A number or symbol representing a number that is
multiplied with a variable or an unknown quantity in an algebraic term.

<a id="parametric"></a>
**[Parametric][#parametric]**: Relating to or expressed in terms of one or more
parameters.

<a id="parametric-methods"></a>
**[Parametric Methods][#parametric-methods]**: A class of techniques that make
explicit assumptions about the shape of the function being modeled and seek to
estimate the assumed function by estimating parameters or coefficients that
yield a function that fits the training data points as closely as possible,
within the constraints of the assumed functional form. Parametric methods tend
to have higher bias since they make assumptions about the form of the function
being modeled.

<a id="polynomial-regression"></a>
**[Polynomial Regression][#polynomial-regression]**: An extension to the linear
model intended to accommodate non-linear relationships and mitigate the effects
of the linear assumption by incorporating polynomial functions of the predictors
into the linear regression model.

For example, in a scenario where a quadratic relationship seems likely, the
following model could be used

$$ \normalsize Y_{i} = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{1}^{2} +
\epsilon $$

<a id="population-regression-line"></a>
**[Population Regression Line][#population-regression-line]**: The line that
describes the best linear approximation to the true relationship between $$ X $$
and $$ Y $$ for the population.

<a id="posterior-probability"></a>
**[Posterior Probability][#posterior-probability]**: Taking into account the
predictor value for a given observation, the probability that the observation
belongs to the kth class of a qualitative variable $$ Y $$ that can take on $$ K
\geq 2 $$ distinct, unordered values. More formally,

$$ p_{k}(x) = \mathrm{Pr}(Y=k|X) . $$

<a id="prediction-interval"></a>
**[Prediction Interval][#prediction-interval]**: A measure of confidence in the
prediction of an individual response, $$ y = f(x) + \epsilon . $$ Prediction
intervals will always be wider than [confidence intervals][#confidence-interval]
because they take into account the uncertainty associated with $$ \epsilon $$,
the irreducible error.

<a id="prior-probability"></a>
**[Prior Probability][#prior-probability]**: The probability that a given
observation is associated with the kth class of a qualitative variable $$ Y $$
that can take on $$ K \geq 2 $$ distinct, unordered values.

<a id="quadratic-discriminant-analysis"></a>
**[Quadratic Discriminant Analysis][#quadratic-discriminant-analysis]**: An
alternative approach to linear discriminant analysis that makes most of the same
assumptions, except that quadratic discriminant analysis assumes that each class
has its own covariance matrix. This amounts to assuming that an observation from
the kth class has a distribution of the form

$$ \normalsize X \sim N(\mu_{k}, \Sigma_{k}) $$

where $$ \Sigma_{k} $$ is a covariance matrix for class $$ k $$.

This yields a Bayes classifier that assigns an observation $$ X = x $$ to the
class with the largest value for

$$ \normalsize \delta_{k}(x) = - \frac{1}{2}(x - \mu_{k})^{T} \Sigma_{k}^{-1} (x
- \mu_{k}) - \frac{1}{2} \log |\Sigma_{k}| + log \pi_{k} $$

which is equivalent to

$$ \normalsize \delta_{k}(x) = - \frac{1}{2}x^{T} \Sigma_{k}^{-1} + x^{T}
\Sigma_{k}^{-1}\mu_{k} - \frac{1}{2}\mu_{k}^{T} \Sigma_{k}^{-1} \mu_{k} -
\frac{1}{2} \log | \Sigma_{k} | + \log \pi_{k} . $$

The quadratic discriminant analysis Bayes classifier gets its name from the fact
that it is a quadratic function in terms of $$ x . $$


<a id="qualitative-value"></a>
**[Qualitative Value][#qualitative-value]**: A value expressed or expressible as
a quality, typically limited to one of K different classes or categories.

<a id="quantitative-value"></a>
**[Quantitative Value][#quantitative-value]**: A value expressed or expressible
as a numerical quantity.

<a id="r-squared-statistic"></a>
**[$$ R^{2} $$ Statistic][#r-squared-statistic]**: A ratio capturing the
proportion of variance explained as a value between $$ 0 $$ and $$
1 $$, independent of the unit of $$ Y . $$

To calculate the $$ R^2 $$ statistic, the following formula may be used

$$ \normalsize R^{2} = \frac{\mathrm{TSS}-\mathrm{RSS}}{\mathrm{TSS}} = 1 -
\frac{\mathrm{RSS}}{\mathrm{TSS}} $$

where RSS is the [residual sum of squares][#residual-sum-of-squares],

$$ \normalsize \mathrm{RSS} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2} , $$

and TSS is the [total sum of squares][#total-sum-of-squares],

$$ \normalsize \mathrm{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y}_{i})^{2} . $$

An $$ R^{2} $$ statistic close to $$ 1 $$
indicates that a large portion of the variability in the response is explained
by the model. An $$ R^{2} $$ value near $$ 0 $$ indicates that the model
accounted for very little of the variability of the model.

An $$ R^{2} $$ value near $$ 0 $$ may occur because the type of model is wrong
and/or because the inherent $$ \sigma^{2} $$ is high.

<a id="regression-problem"></a>
**[Regression Problem][#regression-problem]**: A class of problem that is well
suited to statistical techniques for predicting the value of a dependent
variable or response by modeling a function of one or more independent variables
or predictors in the presence of an error term.

<a id="resampling-methods"></a>
**[Resampling Methods][#resampling-methods]**: Processes of repeatedly drawing
samples from a data set and refitting a given model on each sample with the goal
of learning more about the fitted model.

Resampling methods can be expensive since they require repeatedly performing the
same statistical methods on $$ N $$ different subsets of the data.

<a id="residual"></a>
**[Residual][#residual]**: A quantity left over at the end of a process, a
remainder or excess.

<a id="residual-plot"></a>
**[Residual Plot][#residual-plot]**: A graphical tool useful for identifying
non-linearity. For simple linear regression this consists of graphing the
residuals, $$ e_{i} = y_{i} - \hat{y}_{i} $$ versus the predicted or fitted
values of $$ \hat{y}_{i} .  $$

If a residual plot indicates non-linearity in the model, then a simple approach
is to use non-linear transformations of the predictors, such as $$ \log{x} $$,
$$ \sqrt{x} $$, or $$ x^{2} $$, in the regression model.

![Residual plot for linear and quadratic fits of same data set][graph-residual-plot]

The example residual plots above suggest that a quadratic fit may be more
appropriate for the model under scrutiny.

<a id="residual-standard-error"></a>
**[Residual Standard Error][#residual-standard-error]**: An estimate of standard
error derived from the residual sum of squares, calculated with the following
formula

$$ \normalsize RSE = \sqrt{\frac{\mathrm{RSS}}{n - p - 1}} $$

which simplifies to the following for simple linear regression

$$ \normalsize RSE = \sqrt{\frac{\mathrm{RSS}}{n - 2}} $$

where $$ \mathrm{RSS} $$ is the residual sum of squares. Expanding $$
\mathrm{RSS} $$ yields the formula

$$ \sqrt{\frac{1}{n-p-1}\sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2}} . $$

In rough terms, the residual standard error is the average amount by which the
response will deviate from the true regression line. This means that the
residual standard error amounts to an estimate of the standard deviation of $$
\epsilon $$, the irreducible error. The residual standard error can also be
viewed as a measure of the lack of fit of the model to the data.

<a id="residual-sum-of-squares"></a>
**[Residual Sum of Squares][#residual-sum-of-squares]**: The sum of the
[residual][#residual] square differences between the $$ i \text{th} $$ observed
value and the $$ i \text{th} $$ predicted value.

Assuming the $$ i \text{th} $$ prediction of $$ Y $$ is described as

$$ \normalsize \hat{y_{i}} = \hat{\beta_{0}} + \hat{\beta_{1}}x_{i} $$

then the $$ i \text{th} $$ residual can be represented as

$$ \normalsize e_{i} = y_{i} - \hat{y_{i}} = y_{i} - \hat{\beta_{0}} -
\hat{\beta_{1}}x_{i} . $$

The residual sum of squares can then be described as

$$ \normalsize RSS = e_{1}^2 + e_{2}^2 + \ldots + e_{n}^2 $$

or

$$ \normalsize RSS = (y_{1} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{1})^2 + (y_{2}
- \hat{\beta_{0}} - \hat{\beta_{1}}x_{2})^2 + \ldots + (y_{n} - \hat{\beta_{0}}
- \hat{\beta_{1}}x_{n})^2 .$$

<a id="roc-curve"></a>
**[ROC Curve][#roc-curve]**: A useful graphic for displaying specificity and
sensitivity error rates for all possible posterior probability thresholds. ROC
is a historic acronym that comes from communications theory and stands for
receiver operating characteristics.

![Example ROC curve][roc-curve]

The overall performance of a classifier summarized over all possible thresholds
is quantified by the area under the ROC curve.

A more ideal ROC curve will hold more tightly to the top left corner which, in
turn, will increase the area under the ROC curve. A classifier that performs no
better than chance will have an area under the ROC curve less than or equal to
0.5 when evaluated against a test data set.

<a id="sensitivity"></a>
**[Sensitivity][#sensitivity]**: The percentage of observations correctly
positively classified (true positives).

<a id="simple-linear-regression"></a>
**[Simple Linear Regression][#simple-linear-regression]**: A model that predicts
a quantitative response $$ Y $$ on the basis of a single predictor variable $$ X
.  $$ It assumes an approximately linear relationship between $$ X $$ and $$ Y .
$$ Formally,

$$ \normalsize Y \approx \beta_{0} + \beta_{1}X $$

where $$ \beta_{0} $$ represents the [intercept][#intercept] or the value of $$
Y $$ when $$ X $$ is equal to $$ 0 $$ and $$ \beta_{1} $$ represents the
[slope][#slope] of the line or the average amount of change in $$ Y $$ for each
one-unit increase in $$ X . $$

<a id="slope"></a>
**[Slope][#slope]**: In a linear model, the average amount of change in the
dependent variable, $$ Y , $$ for each one-unit increase in the dependent
variable, $$ X . $$

<a id="specificity"></a>
**[Specificity][#specificity]**: The percentage of observations correctly
negatively classified (true negatives).

<a id="standard-error"></a>
**[Standard Error][#standard-error]**: Roughly, describes the average amount
that an estimate, $$ \hat{\mu} , $$ differs from the actual value of $$ \mu . $$

Standard error can be useful for estimating the accuracy of a single estimated
value, such as an average.  To calculate the standard error of an estimated
value $$ \hat{\mu} $$, the following equation can be used:

$$ \normalsize \mathrm{Var}(\hat{\mu}) = \mathrm{SE}(\hat{\mu})^2 =
\frac{\sigma^{2}}{n} $$

where $$ \sigma $$ is the standard deviation of each the observed values.

The more observations, the larger $$ n $$, the smaller the standard error.

<a id="studentized-residual"></a>
**[Studentized Residual][#studentized-residual]**: Because the standard
deviation of residuals in a sample can vary greatly from one data point to
another even when the errors all have the same standard deviation, it often does
not make sense to compare residuals at different data points without first
studentizing. Studentized residuals are computed by dividing each residual, $$
e_{i} $$, by its estimated standard error. Observations whose studentized
residual is greater than $$ |3| $$ are possible outliers.

<a id="supervised-learning"></a>
**[Supervised Learning][#supervised-learning]**: The process of inferring or
modeling a function from a set of training observations where each observation
consists of one or more features or predictors paired with a response
measurement and the response measurement is used to guide the process of
generating a model that relates the predictors to the response with the goal of
accurately predicting future observations or of better inferring the
relationship between the predictors and the response.

<a id="t-distribution"></a>
**[T-Distribution][#t-distribution]**: Any member of the family of continuous
probability distributions that arise when estimating the mean of a normally
distributed population in situations where the sample size is small and
population standard deviation is unknown.

<a id="t-statistic"></a>
**[T-Statistic][#t-statistic]**: A measure of the number of standard deviations
a quantity a quantity is from zero.

<a id="test-mean-squared-error"></a>
**[Test Mean Squared Error][#test-mean-squared-error]**: The [mean
squared error][#mean-squared-error] yielded when comparing a model's predictions
to the observed values from a previously unseen data set that was not used to
train the model.

<a id="total-sum-of-squares"></a>
**[Total Sum of Squares][#total-sum-of-squares]**: A measure of the total
variance in the response $$ Y . $$ The total sum of squares can be thought of as
the total variability in the response before applying linear regression. The
total sum of squares can be calculated as

$$ \normalsize \mathrm{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y}_{i})^{2} . $$

<a id="training-error-rate"></a>
**[Training Error Rate][#training-error-rate]**: The proportion of
classification errors that are made when applying $$ \hat{f} $$ to the training
observations. Formally stated as,

$$ \normalsize \frac{1}{n} \sum_{i=1}^{n} \mathrm{I}(y_{i} \neq \hat{y}) $$

where $$ \mathrm{I} $$ is an indicator variable that equals $$ 0 $$ when $$ y =
\hat{y} $$ and equals $$ 1 $$ when $$ y \neq \hat{y} . $$

In simple terms, the training error rate is the ratio of incorrect
classifications to the count of training observations.

<a id="training-mean-squared-error"></a>
**[Training Mean Squared Error][#training-mean-squared-error]**: The [mean
squared error][#mean-squared-error] yielded when comparing a model's predictions
to the observed values that were used to train that same model.

<a id="unbiased-estimator"></a>
**[Unbiased Estimator][#unbiased-estimator]**: A model that does not
systematically overestimate or underestimate when generating predictions of the
true response value.

<a id="unsupervised-learning"></a>
**[Unsupervised Learning][#unsupervised-learning]**: The process of inferring or
modeling a function from a set of training observations where each observation
consists of one or more features or predictors. Unlike supervised learning, no
response measurement is available to supervise the analysis that goes into
generating the model.

<a id="validation-set"></a>
**[Validation Set][#validation-set]**: A randomly selected subset of a data set
that is withheld for the purpose of validating model fit and estimating test
error rate.

Though conceptually simple and easy to implement, the validation set approach
has two potential drawbacks.

1. The estimated test error rate can be highly variable depending on which
observations fall into the training set and which observations fall into the
test/validation set.

2. The estimated error rate tends to be overestimated since the given
statistical method was trained with fewer observations than it would have if
fewer observations had been set aside for validation.

[Cross-validation][#cross-validation] is a refinement of the validation set
approach that mitigates these two issues.

<a id="variable-selection"></a>
**[Variable Selection][#variable-selection]**: The process of removing
extraneous predictors that don’t relate to the response.

<a id="variance"></a>
**[Variance][#variance]**: The amount by which $$ \hat{f} $$ would change or vary
if it were estimated using a different training data set. In general, more
flexible methods have higher variance.

<a id="variance-inflation-factor"></a>
**[Variance Inflation Factor][#variance-inflation-factor]**: The ratio of the
variance of $$ \hat{\beta}_{j} $$ when fitting the full model divided by the
variance of $$ \hat{\beta}_{j} $$ if fit on its own. The smallest possible
variance inflation factor value is $$ 1.0 $$, which indicates no
[collinearity][#collinearity] whatsoever. In practice, there is typically a
small amount of collinearity among predictors. As a general rule of thumb,
variance inflation factor values that exceed 5 or 10 indicate a problematic
amount of collinearity.

The variance inflation factor for each variable can be computed using the
formula

$$ \normalsize \mathrm{VIF}(\hat{\beta_{j}}) = \frac{1}{1 -
\mathrm{R}_{x_{j}|x_{-j}}^{2}} $$

where $$ \mathrm{R}_{x_{j}|x_{-j}} $$ is the $$ \mathrm{R}^{2} $$ from a
regression of $$ X_{j} $$ onto all of the other predictors. If $$
\mathrm{R}_{x_{j}|x_{-j}} $$ is close to one, the VIF will be large and
collinearity is present.

<a id="z-statistic"></a>
**[Z-Statistic][#z-statistic]**: Similar to the [t-statistic][#t-statistic],
logistic regression measures the accuracy of coefficient estimates using a
quantity called the z-statistic.  The z-statistic for $$ \beta_{1} $$ is
represented by

$$ \normalsize \textrm{z-statistic}(\beta_{1}) =
\frac{\hat{\beta}_{1}}{\mathrm{SE}(\hat{\beta}_{1})} $$

A large z-statistic offers evidence against the null hypothesis.

[#backwards-selection]: #backwards-selection "Backwards Selection"
[#bayes-classifier]: #bayes-classifier "Bayes Classifier"
[#bayes-decision-boundary]: #bayes-decision-boundary "Bayes Decision Boundary"
[#bayes-error-rate]: #bayes-error-rate "Bayes Error Rate"
[#bayes-theorem]: #bayes-theorem "Bayes Theorem"
[#bias]: #bias "Bias"
[#bias-variance-trade-off]: #bias-variance-trade-off "Bias-Variance Trade-Off"
[#bootstrap]: #bootstrap "Bootstrap"
[#classification-problem]: #classification-problem "Classification Problem"
[#cluster-analysis]: #cluster-analysis "Cluster Analysis"
[#coefficient]: #coefficient "Coefficient"
[#collinearity]: #collinearity "Collinearity"
[#confidence-interval]: #confidence-interval "Confidence Interval"
[#confounding]: #confounding "Confounding"
[#correlation]: #correlation "Correlation"
[#cross-validation]: #cross-validation "Cross Validation"
[#curse-of-dimensionality]: #curse-of-dimensionality "Curse of Dimensionality"
[#degrees-of-freedom]: #degrees-of-freedom "Degrees of Freedom"
[#density-function]: #density-function "Density Function"
[#discriminant-analysis]: #discriminant-analysis "Discriminant Analysis"
[#dummy-variable]: #dummy-variable "Dummy Variable"
[#error-term]: #error-term "Error Term"
[#f-distribution]: #f-distribution "F-Distribution"
[#f-statistic]: #f-statistic "F-Statistic"
[#forward-selection]: #forward-selection "Forward Selection"
[#gaussian-distribution]: #gaussian-distribution "Gaussian Distribution"
[#heteroscedasticity]: #heteroscedasticity "Heteroscedasticity"
[#hierarchical-principle]: #hierarchical-principle "Hierarchical Principle"
[#high-leverage]: #high-leverage "High Leverage"
[#hypothesis-testing]: #hypothesis-testing "Hypothesis Testing"
[#input]: #input "Input"
[#indicator-variable]: #indicator-variable "Indicator Variable"
[#interaction-term]: #interaction-term "Interaction Term"
[#intercept]: #intercept "Intercept"
[#irreducible-error]: #irreducible-error "Irreducible Error"
[#k-fold-cross-validation]: #k-fold-cross-validation "K-Fold Cross Validation"
[#k-nearest-neighbors-classifier]: #k-nearest-neighbors-classifier "K-Nearest Neighbors Classifier"
[#k-nearest-neighbors-regression]: #k-nearest-neighbors-regression "K-Nearest Neighbors Regression"
[#least-squares-line]: #least-squares-line "Least Squares Line"
[#leave-one-out-cross-validation]: #leave-one-out-cross-validation "Leave One Out Cross Validation"
[#likelihood-function]: #likelihood-function "Likelihood Function"
[#linear-discriminant-analysis]: #linear-discriminant-analysis "Linear Discriminant Analysis"
[#log-odds]: #log-odds "Log-Odds"
[#logistic-function]: #logistic-function "Logistic Function"
[#logistic-regression]: #logistic-regression "Logistic Regression"
[#logit]: #logit "Logit"
[#maximum-likelihood]: #maximum-likelihood "Maximum Likelihood"
[#mean-squared-error]: #mean-squared-error "Mean Squared Error"
[#mixed-selection]: #mixed-selection "Mixed Selection"
[#model-assessment]: #model-assessment "Model Assessment"
[#model-selection]: #model-selection "Model Selection"
[#multicollinearity]: #multicollinearity "Multicollinearity"
[#multiple-linear-regression]: #multiple-linear-regression "Multiple Linear Regression"
[#multiple-logistic-regression]: #multiple-logistic-regression "Multiple Logistic Regression"
[#multivariate-gaussian-distribution]: #multivariate-gaussian-distribution "Multivariate Gaussian Distribution"
[#normal-distribution]: #normal-distribution "Normal Distribution"
[#non-parametric]: #non-parametric "Non-Parametric"
[#non-parametric-methods]: #non-parametric-methods "Non-Parametric Methods"
[#null-hypothesis]: #null-hypothesis "Null Hypothesis"
[#null-model]: #null-model "Null Model"
[#odds]: #odds "Odds"
[#output]: #output "Output"
[#outlier]: #outlier "Outlier"
[#overfitting]: #overfitting "Overfitting"
[#p-value]: #p-value "P-Value"
[#parameter]: #parameter "Parameter"
[#parametric]: #parametric "Parametric"
[#parametric-methods]: #parametric-methods "Parametric Methods"
[#polynomial-regression]: #polynomial-regression "Polynomial Regression"
[#population-regression-line]: #population-regression-line "Population Regression Line"
[#posterior-probability]: #posterior-probability "Posterior Probability"
[#prediction-interval]: #prediction-interval "Prediction Interval"
[#prior-probability]: #prior-probability "Prior Probability"
[#quadratic-discriminant-analysis]: #quadratic-discriminant-analysis "Quadratic Discriminant Analysis"
[#qualitative-value]: #qualitative-value "Qualitative Value"
[#quantitative-value]: #quantitative-value "Quantitative Value"
[#r-squared-statistic]: #r-squared-statistic "R Squared Statistic"
[#regression-problem]: #regression-problem "Regression Problem"
[#resampling-methods]: #resampling-methods "Resampling Methods"
[#residual]: #residual "Residual"
[#residual-plot]: #residual-plot "Residual Plot"
[#residual-standard-error]: #residual-standard-error "Residual Standard Error"
[#residual-sum-of-squares]: #residual-sum-of-squares "Residual Sum of Squares"
[#roc-curve]: #roc-curve "ROC Curve"
[#sensitivity]: #sensitivity "Sensitivity"
[#simple-linear-regression]: #simple-linear-regression "Simple Linear Regression"
[#slope]: #slope "Slope"
[#specificity]: #specificity "Specificity"
[#standard-error]: #standard-error "Standard Error"
[#studentized-residual]: #studentized-residual "Studentized Residual"
[#supervised-learning]: #supervised-learning "Supervised Learning"
[#test-mean-squared-error]: #test-mean-squared-error "Test Mean Squared Error"
[#total-sum-of-squares]: #total-sum-of-squares "Total Sum of Squares"
[#training-error-rate]: #training-error-rate "Training Error Rate"
[#training-mean-squared-error]: #training-mean-squared-error "Training Mean Squared Error"
[#t-distribution]: #t-distribution "T-Distribution"
[#t-statistic]: #t-statistic "T-Statistic"
[#unsupervised-learning]: #unsupervised-learning "Unsupervised Learning"
[#unbiased-estimator]: #unbiased-estimator "Unbiased Estimator"
[#validation-set]: #validation-set "Validation Set"
[#variable-selection]: #variable-selection "Variable Selection"
[#variance]: #variance "Variance"
[#variance-inflation-factor]: #variance-inflation-factor "Variance Inflation Factor"
[#z-statistic]: #z-statistic "Z-Statistic"
[graph-residual-plot]: images/residual-plot.jpg "Residual plots for linear and quadratic fits of same data set"
[roc-curve]: images/ROC-curve.jpg "Example ROC curve"
