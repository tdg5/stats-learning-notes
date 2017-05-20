---
layout: default
use_math: true
---

[Previous: Chapter 3 - Linear Regression][chapter-03-linear-regression]

---

## [Chapter 4 - Classification][chapter-04-classification]

Linear regression could be applied in the case of binary qualitative responses,
but beyond two levels, choosing a coding scheme is problematic and different
coding scheme can yield wildly different predictions.

### Logistic Regression

[Logistic regression][glossary-logistic-regression] models the probability that
$$ y $$ belongs to a particular category rather than modeling the response
itself.

Logistic regression uses the [logistic function][glossary-logistic-function] to
ensure a prediction between $$ 0 $$ and $$ 1 .$$ The logistic function takes the
form

$$ \normalsize p(X) = \frac{e^{\beta_{0} + \beta_{1}X}}{1 + e^{\beta_{0} +
\beta_{1}X}} .$$

This yields a probability greater than $$ 0 $$ and less than $$ 1 .$$

The logistic function can be rebalanced to yield

$$ \normalsize \frac{p(X)}{1 - p(X)} = e^{\beta_{0} + \beta_{1}X} $$

$$ \frac{p(X)}{1 - p(X)} $$ is known as the odds and takes on a value between
$$ 0 $$ and infinity.

As an example, a probability of 1 in 5 yields odds of $$ \frac{1}{4} $$ since
$$ \frac {0.2}{1 - 0.2} = \frac{1}{4} .$$

Taking a logarithm of both sides of the logistic odds equation yields an
equation for the [log-odds][glossary-log-odds] or [logit][glossary-logit],

$$ \normalsize \mathrm{log} \bigg \lgroup \frac{p(X)}{1 - p(X)} \bigg \rgroup =
\beta_{0} + \beta_{1}X $$

Logistic regression has a logit that is linear in terms of $$ X . $$

Unlike linear regression where $$ \beta_{1} $$ represents the average change in
$$ Y $$ with a one-unit increase in $$ X $$, for logistic regression, increasing
$$ X $$ by one-unit yields a $$ \beta_{1} $$ change in the log-odds which is
equivalent to multiplying the odds by $$ e^{\beta_{1}} .$$

The relationship between $$ p(X) $$ and $$ X $$ is not linear and because of
this $$ \beta_{1} $$ does not correspond to the change in $$ p(X) $$ given
one-unit increase in $$ X $$. However, if $$ \beta_{1} $$ is positive,
increasing $$ X $$ will be associated with an increase in $$ p(X) $$ and,
similarly, if $$ \beta_{1} $$ is negative, an increase in $$ X $$ will be
associated with a decrease in $$ p(X) $$. How much change will depend on the
value of $$ X $$.

#### Estimating Regression Coefficients

Logistic regression uses a strategy called [maximum
likelihood][glossary-maximum-likelihood] to estimate regression coefficients.

Maximum likelihood plays out like so, determine estimates for $$ \beta_{0} $$
and $$ \beta_{1} $$ such that the predicted probability of $$ \hat{p}(x_{i}) $$
corresponds with the observed classes as closely as possible. Formally, this
yield an equation called a [likelihood function][glossary-likelihood-function]:

$$ \normalsize \ell(\beta_{0}, \beta_{1}) = \displaystyle
\prod_{i:y_{i}=1}p(X_{i}) \times \displaystyle \prod_{j:y_{j}=0}(1-p(X_{j})) .
$$

Estimates for $$ \beta_{0} $$ and $$ \beta_{1} $$ are chosen so as to maximize
this likelihood function.

Linear regression's least squares approach is actually a special case of maximum
likelihood.

Logistic regression measures the accuracy of coefficient estimates using a
quantity called the [z-statistic][glossary-z-statistic]. The z-statistic is
similar to the t-statistic. The z-statistic for $$ \beta_{1} $$ is represented
by

$$ \normalsize \textrm{z-statistic}(\beta_{1}) =
\frac{\hat{\beta}_{1}}{\mathrm{SE}(\hat{\beta}_{1})} $$

A large z-statistic offers evidence against the null hypothesis.

In logistic regression, the null hypothesis

$$ \normalsize H_{0}: \beta_{1} = 0 $$

implies that

$$ \normalsize p(X) = \frac{e^{\beta_{0}}}{1 + e^{\beta_{0}}} $$

and, ergo, $$ p(X) $$ does not depend on $$ X . $$

#### Making Predictions

Once coefficients have been estimated, predictions can be made by plugging the
coefficients into the model equation

$$ \normalsize \hat{p}(X) = \frac{e^{\hat{\beta_{0}} + \hat{\beta_{1}}X}}{1 +
e^{\hat{\beta}_{0} + \hat{\beta}_{1}X}} . $$

In general, the estimated intercept, $$ \hat{\beta}_{0} , $$ is of limited
interest since it mainly captures the ratio of positive and negative
classifications in the given data set.

Similar to linear regression, dummy variables can be used to accommodate
qualitative predictors.

### Multiple Logistic Regression

Using a strategy similar to that employed for linear regression, [multiple
logistic regression][glossary-multiple-logistic-regression] can be generalized
as

$$ \normalsize log \bigg \lgroup \frac{p(X)}{1 - p(X)} \bigg \rgroup = \beta_{0}
+ \beta_{1}X_{1} + \ldots + \beta_{p}X_{p} $$

where $$ X = (X_{1}, X_{2}, \ldots, X_{p}) $$ are $$ p $$ predictors.

The log-odds equation for multiple logistic regression can be expressed as

$$ \normalsize p(X) = \frac{e^{\beta_{0} + \beta_{1}X_{1} + \ldots +
\beta_{p}X_{p}}}{1 + e^{\beta_{0} + \beta_{1}X_{1} + \ldots + \beta_{p}X_{p}}} $$

Maximum likelihood is also used to estimate $$ \beta_{0}, \beta_{1}, \ldots,
\beta_{p} $$ in the case of multiple logistic regression.

---

[Next: Chapter 5 - Resampling Methods][chapter-05-resampling-methods]

<a id="bottom"></a>

[chapter-03-linear-regression]: chapter-03-linear-regression "stats-learning-notes -- Chapter 3 - Linear Regression"
[chapter-04-classification]: chapter-04-classification "stats-learning-notes -- Chapter 4 - Classification"
[chapter-05-resampling-methods]: chapter-05-resampling-methods "stats-learning-notes -- Chapter 5 - Resampling Methods"
[glossary-likelihood-function]: glossary#likelihood-function "stats-learning-notes -- Glossary - Likelihood Function"
[glossary-log-odds]: glossary#log-odds "stats-learning-notes -- Glossary - Log-Odds"
[glossary-logistic-function]: glossary#logistic-function "stats-learning-notes -- Glossary - Logistic Function"
[glossary-logistic-regression]: glossary#logistic-regression "stats-learning-notes -- Glossary - Logistic Regression"
[glossary-logit]: glossary#logit "stats-learning-notes -- Glossary - Logit"
[glossary-maximum-likelihood]: glossary#maximum-likelihood "stats-learning-notes -- Glossary - Maximum Likelihood"
[glossary-multiple-logistic-regression]: glossary#multiple-logistic-regression "stats-learning-notes -- Glossary - Multiple Logistic Regression"
