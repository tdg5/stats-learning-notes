---
layout: default
use_math: true
---

## [Chapter 3 - Linear Regression][jekyll-site-chapter-3]

### Simple Linear Regression

[Simple linear regression][glossary-simple-linear-regression] predicts a
quantitative response $$ Y $$ on the basis of a single predictor variable $$ X
$$. It assumes an approximately linear relationship between $$ X $$ and $$ Y $$.
Formally,

$$ Y \approx \beta_{0} + \beta_{1}X $$

where $$ \beta_{0} $$ represents the [intercept][glossary-intercept] or the
value of $$ Y $$ when $$ X $$ is equal to $$ 0 $$ and $$ \beta_{1} $$ represents the
[slope][glossary-slope] of the line or the average amount of change in $$ Y $$
for each one-unit increase in $$ X $$.

Together, $$ \beta_{0} $$ and $$ \beta_{1} $$ are known as the model
[coefficients][glossary-coefficient] or [parameters][glossary-parameter].

#### Estimating Model Coefficients

Since $$ \beta_{0} $$ and $$ \beta_{1} $$ are typically unknown, it is first
necessary to estimate the coefficients before making predictions. To estimate
the coefficients, it is desirable to choose values for $$ \beta_{0} $$ and $$
\beta_{1} $$ such that the resulting line is as close as possible to the
observed data points.

There are many ways of measuring closeness. The most common method strives to
minimizes the sum of the [residual][glossary-residual] square differences
between the $$ i $$th observed value and the $$ i $$th predicted value.

Assuming the $$ i $$th prediction of $$ Y $$ is described as

$$ \hat{y_{i}} = \hat{\beta_{0}} + \hat{\beta_{1}}x_{i} $$

then the $$ i $$th residual can be represented as

$$ e_{i} = y_{i} - \hat{y_{i}} = y_{i} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{i} . $$

The [residual sum of squares][glossary-residual-sum-of-squares] can then be
described as

$$ RSS = e_{1}^2 + e_{2}^2 + \ldots + e_{n}^2 $$

or

$$ RSS = (y_{1} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{1})^2 + (y_{2} -
\hat{\beta_{0}} - \hat{\beta_{1}}x_{2})^2 + \ldots + (y_{n} - \hat{\beta_{0}} -
\hat{\beta_{1}}x_{n})^2 .$$

Assuming sample means of

$$ \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_{i} $$

and

$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_{i} , $$

calculus can be applied to estimate the least squares coefficient estimates for
linear regression to minimize the residual sum of squares like so

$$ \beta_{1} = \frac{\sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} -
\bar{y})}{\sum_{i=1}^{n}(x_{i} - \bar{x})^2} $$

$$ \beta_{0} = \bar{y} - \hat{\beta_{1}}\bar{x} $$

#### Assessing Coefficient Estimate Accuracy

Simple linear regression represents the relationship between $$ Y $$ and $$ X $$
as

$$ Y = \beta_{0} + \beta_{1}X + \epsilon $$

where

$$ \beta_{0} $$ is the intercept term, or the value of $$ Y $$ when $$ X = 0 $$

$$ \beta_{1} $$ is the slope, or average increase in $$ Y $$ associated with a
one-unit increase in $$ X $$

$$ \epsilon $$ is the error term which acts as a catchall for what is missed by
the simple model: the true relationship likely isn't linear; there may be other
variables that affect $$ Y $$; there may be error in the observed measurements.
The error term is typically assumed to be independent of $$ X $$.

The model used by simple linear regression defines the [population
regression line][glossary-population-regression-line], which describes the best
linear approximation to the true relationship between $$ X $$ and $$ Y $$ for
the population.

The least squares regression coefficient estimates characterize the [least
squares line][glossary-least-squares-line],

$$ \hat{y_{i}} = \hat{\beta_{0}} + \hat{\beta_{1}}x_{i} . $$

The difference between the population regression line and the least squares
line is similar to using a sample to estimate the characteristics of a large
population.

In linear regression, the unknown coefficients, $$ \beta_{0} $$ and $$ \beta_{1}
$$ define the population regression line, whereas the estimates of those
coefficients, $$ \hat{\beta_{0}} $$ and $$ \hat{\beta_{1}} $$ define the least
squares line.

Though the parameter estimates for a given sample may overestimate or
underestimate the value of a particular parameter, an unbiased estimator does
not systemically overestimate or underestimate the true parameter.

This means that using an unbiased estimator and a large number of data sets, the
values of the coefficients $$ \beta_{0} $$ and $$ \beta_{1} $$ could be
determined by averaging the coefficient estimates from each of those data sets.

To estimate the accuracy of a single estimated value, such as an average, it can
be helpful to calculate the [standard error][glossary-standard-error] of the
estimated value $$ \hat{\mu} $$, which can be accomplished like so:

$$ \mathrm{Var}(\hat{\mu}) = \mathrm{SE}(\hat{\mu})^2 = \frac{\sigma^{2}}{n} $$

where $$ \sigma $$ is the standard deviation of each $$ y_{i} $$.

Roughly, the standard error describes the average amount that the estimate $$
\hat{\mu} $$ differs from $$ \mu $$.

The more observations, the larger $$ n $$, the smaller the standard error.

To compute the standard errors associated with $$ \beta_{0} $$ and $$ \beta_{1}
$$, the following formulas can be used:

$$ \mathrm{SE}(\beta_{0})^{2} = \sigma^{2}[\frac{1}{n} +
\frac{\bar{x}^{2}}{\sum_{i=1}^{n}(x_{i} - \bar{x})^2}] $$

and

$$ \mathrm{SE}(\beta_{1})^{2} = \frac{\sigma^{2}}{\sum_{i=1}^{n}(x_{i} - \bar{x})^2} $$

where $$ \sigma^{2} = \mathrm{Var}(\epsilon) $$ and $$ \epsilon_{i} $$ is not
correlated with $$ \sigma^{2} $$.

$$ \sigma^{2} $$ generally isn't known, but can be estimated from the data. The
estimate of $$ \sigma $$ is known as the [residual standard
error][glossary-residual-standard-error] and can be calculated with the
following formula

$$ \mathrm{RSE} = \sqrt{\frac{\mathrm{RSS}}{(n - 2)}} . $$

where $$ \mathrm{RSS} $$ is the residual sum of squares.

Standard errors can be used to compute confidence intervals. A [confidence
interval][glossary-confidence-interval] is defined as a range of values such
that there's a certain likelihood that the range will contain the true unknown
value of the parameter.

For simple linear regression the 95% confidence interval for $$ \beta_{1} $$ can be
approximated by

$$ \hat{\beta_{1}} \pm 2 \times \mathrm{SE}(\hat{\beta_{1}}) . $$

Similarly, a confidence interval for $$ \beta_{0} $$ can be approximated as

$$ \hat{\beta_{0}} \pm 2 \times \mathrm{SE}(\hat{\beta_{0}}) $$

The accuracy of an estimated prediction depends on whether we wish to predict an
individual response, $$ y = f(x) + \epsilon $$, or the average response, $$ f(x)
$$.

When predicting an individual response, $$ y = f(x) + \epsilon $$, a prediction
interval is used.

When predicting an average response, $$ f(x) $$, a confidence interval is used.

Prediction intervals will always be wider than confidence intervals because they
take into account the uncertainty associated with $$ \epsilon $$, the
irreducible error.

The standard error can also be used to perform [hypothesis
tests][glossary-hypothesis-test] on the estimated coefficients.

The most common hypothesis test involves testing the [null
hypothesis][glossary-null-hypothesis] that states

$$ H_{0} $$: There is no relationship between $$ X $$ and $$ Y $$

versus the alternative hypothesis

$$ H_{1} $$: Thee is some relationship between $$ X $$ and $$ Y $$.

In mathematical terms, the null hypothesis corresponds to testing if $$
\beta_{1} = 0 $$, which reduces to

$$ Y = \beta_{0} + \epsilon $$

which evidences that $$ X $$ is not related to $$ Y $$.

To test the null hypothesis, it is necessary to determine whether the estimate
of $$ \beta_{1} $$, $$ \hat{\beta_{1}} $$, is sufficiently far from zero to provide
confidence that $$ \beta_{1} $$ is non-zero.

How close is close enough depends on $$ \mathrm{SE}(\hat{\beta_{1}}) $$. When $$
\mathrm{SE}(\hat{\beta_{1}}) $$ is small, then small values of $$
\hat{\beta_{1}} $$ may provide strong evidence that $$ \beta_{1} $$ is not zero.
Conversely, if $$ \mathrm{SE}(\hat{\beta_{1}}) $$ is large, then $$
\hat{\beta_{1}} $$ will need to be large in order to reject the null hypothesis.

In practice, computing a [T-statistic][glossary-t-statistic], which measures the
number of standard deviations that $$ \hat{\beta_{1}} $$, is away from $$ 0 $$,
is useful for determining if an estimate is sufficiently significant to reject
the null hypothesis.

A T-statistic can be computed as follows

$$ t = \frac{\hat{\beta}_{1} - 0}{\mathrm{SE}(\hat{\beta_{1}})} $$

If there is no relationship between $$ X $$ and $$ Y $$, it is expected that a
[t-distribution][glossary-t-distribution] with $$ n - 2 $$ degrees of freedom
should be yielded.

With such a distribution, it is possible to calculate the probability of
observing a value of $$ |t| $$ or larger assuming that $$ \hat{\beta_{1}} = 0
$$. This probability, called the [p-value][glossary-p-value], can indicate an
association between the predictor and the response if sufficiently small.

#### Assessing Model Accuracy

Once the null hypothesis has been rejected, it may be desirable to quantify to
what extent the model fits the data. The quality of a linear regression model is
typically assessed using residual standard error (RSE) and the $$ R^2 $$
statistic.

The residual standard error is an estimate of the standard deviation of $$
\epsilon $$, the irreducible error.

In rough terms, the residual standard error is the average amount by which the
response will deviate from the true regression line.

For linear regression, the residual standard error can be computed as

$$ \mathrm{RSE} = \sqrt{\frac{1}{n-2}\mathrm{RSS}} =
\sqrt{\frac{1}{n-2}\sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2}} $$

The residual standard error is a measure of the lack of fit of the model to the
data. When the values of $$ y_{i} \approx \hat{y}_{i} $$, the RSE will be small
and the model will fit the data well. Conversely, if $$ y_{i} \ne \hat{y_{i}} $$
for some values, the RSE may be large, indiating that the model doesn't fit the
data well.

The RSE provides an absolute measure of the lack of fit of the model in the
units of $$ Y $$. This can make it difficult to know what constitutes a good
RSE.

The [$$ R^{2} $$ statistic][glossary-r-squared-statistic] is an alternative measure of
fit that takes the form of a proportion. The $$ R^{2} $$ statistic captures the
proportion of variance explained as a value between $$ 0 $$ and $$ 1 $$,
independent of the unit of $$ Y $$.

To calculate the $$ R^2 $$ statistic, the following formula may be used

$$ R^{2} = \frac{\mathrm{TSS}-\mathrm{RSS}}{\mathrm{TSS}} = 1 -
\frac{\mathrm{RSS}}{\mathrm{TSS}} $$

where

$$ \mathrm{RSS} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2} $$

and

$$ \mathrm{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y}_{i})^{2} $$

The total sum of squares, TSS, measures the total variance in the response $$ Y
$$. The TSS can be thought of as the total variability in the response before
applying linear regression. Conversely, the residual sum of squares, RSS,
measures the amount of variability left after performing the regression.

Ergo, $$ TSS - RSS $$ measures the amount of variability in the response that is
explained by the model. $$ R^{2} $$ measures the proportion of variability in $$
Y $$ that can be explained by $$ X $$. An $$ R^{2} $$ statistic close to $$ 1 $$
indicates that a large portion of the variability in the response is explained
by the model. An $$ R^{2} $$ near $$ 0 $$ indicates that the model accounted for
very little of the variability of the model.

An $$ R^{2} $$ value near $$ 0 $$ may occur because the linear model is wrong
and/or because the inherent $$ \sigma^{2} $$ is high.

$$ R^{2} $$ has an advantage over RSE since it will always yield a value between $$
0 $$ and $$ 1 $$, but it can still be tough to know what a good $$ R^{2} $$
value is. Frequently, what constitutes a good $$ R^{2} $$ value depends on the
application and what is known about the problem.

The $$ R^{2} $$ statistic is a measure of the linear relationship between $$ X
$$ and $$ Y $$.

[Correlation][glossary-correlation] is another measure of the linear relationship between $$ X $$ and $$
Y $$. Correlation of can be calculated as

$$ \mathrm{Cor}(X,Y) = \frac{\sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} -
\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_{i} -
\bar{x})^{2}}\sqrt{\sum_{i=1}^{n}(y_{i}-\bar{y})^{2}}} $$

This suggests that $$ r = \mathrm{Cor}(X,Y) $$ could be used instead of $$ R^{2}
$$ to assess the fit of the linear model, however for simple linear regression
it can be shown that $$ R^{2} = r^{2} $$. More concisely, for simple linear
regression, the squared correlation and the $$ R^{2} $$ statistic are equivalent. Though this is the case for simple linear regression, correlation
does not extend to multiple linear regression since correlation quantifies the
association between a single pair of variables. $$ R^{2} $$ can, however, be
applied to multiple regression.

### Multiple Regression

The [multiple linear regression][glossary-multiple-linear-regression] model
takes the form of

$$ Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \ldots + \beta_{p}X_{p} +
\epsilon . $$

Multiple linear regression extends simple linear regression to accommodate
multiple predictors.

$$ X_{j} $$ represents the $$ j $$th predictor and $$ \beta_{j} $$ represents
the average effect of a one-unit increase in $$ X_{j} $$ on $$ Y $$, holding all
other predictors fixed.

#### Estimating Multiple Regression Coefficients

Because the coefficients $$ \beta_{0}, \beta_{1}, \beta_{2}, \ldots, \beta_{p} $$
are unknown, it is necessary to estimate their values. Given estimates of $$
\hat{\beta_{0}}, \hat{\beta_{1}}, \hat{\beta_{2}}, \ldots, \hat{\beta_{p}} $$,
estimates can be made using the formula below

$$ \hat{y} = \hat{\beta_{0}} + \hat{\beta_{1}}x_{1} + \hat{\beta_{2}}x_{2} +
\ldots + \hat{\beta_{p}}x_{p} $$

The parameters $$ \hat{\beta_{0}}, \hat{\beta_{1}}, \hat{\beta_{2}}, \ldots,
\hat{\beta_{p}} $$ can be estimated using the same least squares strategy as was
employed for simple linear regression. Values are chosen for the parameters $$
\hat{\beta_{0}}, \hat{\beta_{1}}, \hat{\beta_{2}}, \ldots, \hat{\beta_{p}} $$ such
that the residual sum of squares is minimized

$$ RSS = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2} = \sum_{i=1}^{n}(y_{i} -
\hat{\beta_{0}} - \hat{\beta_{1}}x_{1} - \hat{\beta_{2}}x_{2} - \ldots -
\hat{\beta_{p}}x_{p})^{2} $$

Estimating the values of these parameters is best achieved with matrix algebra.

#### Assessing Multiple Regression Coefficient Accuracy

Once estimates have been derived, it is next appropriate to test the null
hypothesis

$$ H_{0}: \beta_{1} = \beta_{2} = \ldots = \beta_{p} = 0 $$

versus the alternative hypothesis

$$ H_{a}: at\ least\ one\ of B_{j} \ne 0 . $$

The [F-statistic][glossary-f-statistic] can be used to determine which
hypothesis holds true.

The F-statistic can be computed as

$$ \mathrm{F} = \frac{(\mathrm{TSS} - \mathrm{RSS})/p}{\mathrm{RSS}/(n - p - 1)}
= \frac{\frac{\mathrm{TSS} - \mathrm{RSS}}{p}}{\frac{\mathrm{RSS}}{n - p - 1}}
$$

where, again,

$$ \mathrm{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y}_{i})^{2} $$

and

$$ \mathrm{RSS} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^2 $$

If the assumptions of the linear model, represented by the alternative
hypothesis, are true it can be shown that

$$ \mathrm{E}\{\frac{\mathrm{RSS}}{n - p - 1}\} = \sigma^{2} $$

Conversely, if the null hypothesis is true, it can be shown that

$$ \mathrm{E}\{\frac{\mathrm{TSS} - \mathrm{RSS}}{p}\} = \sigma^{2} $$

This means that when there is no relationship between the response and the
predictors the F-statisitic takes on a value close to $$ 1 $$.

Conversely, if the alternative hypothesis is true, then the F-statistic will
take on a value greater than $$ 1 $$.

When $$ n $$ is large, an F-statistic only slightly greater than $$ 1 $$ may
provide evidence against the null hypothesis. If $$ n $$ is small, a large
F-statistic is needed to reject the null hypothesis.

When the null hypothesis is true and the errors $$ \epsilon_{i} $$ have a [normal
distribution][glossary-normal-distribution], the F-statistic follows and
[F-distribution][glossary-f-distribution]. Using the F-distribution, it is
possible to figure out a p-value for the given $$ n $$, $$ p $$, and
F-statistic. Based on the obtained p-value, the validity of the null hypothesis
can be determined.

It is sometimes desirable to test that a particular subset of $$ q $$
coefficients are $$ 0 $$. This equates to a null hypothesis of

$$ H_{0}: \beta_{p - q + 1} = \beta_{p - q + 2} = \ldots = \beta_{p} = 0 . $$

Supposing that the residual sum of squares for such a model is $$
\mathrm{RSS}_{0} $$ then the F-statistic could be calculated as

$$ \mathrm{F} = \frac{(\mathrm{RSS}_{0} - \mathrm{RSS})/q}{\mathrm{RSS}/(n - p -
1)} = \frac{\frac{\mathrm{RSS}_{0} - \mathrm{RSS}}{q}}{\frac{\mathrm{RSS}}{n - p -
1}} . $$

Even in the presence of p-values for each individual variable, it is still
important to consider the overall F-statistic because there is a reasonably high
likelihood that a variable with a small p-value will occur just by chance, even
in the absence of any true association between the predictors and the response.

In contrast, the F-statistic does not suffer from this problem because it
adjusts for the number of predictors. The F-statistic is not infallible and when
the null hypothesis is true the F-statisistc can still result in p-values below
$$ 0.05 $$ about 5% of the time regardless of the number of predictors or the
number of observations.

The F-statistic works best when $$ p $$ is relatively small or when $$ p $$ is
relatively small compared to $$ n $$.

When $$ p $$ is greater than $$ n $$, multiple linear regression using least
squares will not work, and similarly, the F-statistic cannot be used either.

#### Selecting Important Variables

Once it has been established that at least one of the predictors is associated
with the response, the question remains, _which_ of the predictors is related to
the response? The process of removing extraneous predictors that don't relate to
the response is called [variable-selection][glossary-variable-selection].

Ideally, the process of variable selection would involve testing many different
models, each with a different subset of the predictors, then selecting the best
model of the bunch, with the meaning of "best" being derived from various
statistical methods.

Regrettably, there are a total of $$ 2^{p} $$ models that contain subsets of $$
p $$ predictors. Because of this, an efficient an automated means of choosing a
smaller subset of models is needed. There are a number of statistical approaches
to limiting the range of possible models.

[Forward selection][glossary-forward-selection] begins with a [null
model][glossary-null-model], a model that has an intercept but no predictors,
and attempts $$ p $$ simple linear regressions, keeping whichever predictor
results in the lowest residual sum of squares. In this fashion, the predictor
yielding the lowest RSS is added to the model one-by-one until some halting
condition is met. Forward selection is a greedy process and it may include
extraneous variables.

[Backwards selection][glossary-backwards-selection] begins with a model that
includes all the predictors and proceeds by removing the variable with the
highest p-value each iteration until some stopping condition is met. Backwards
selection cannot be used when $$ p > n $$.

[Mixed selection][glossary-mixed-selection] begins with a null model, like
forward selection, repeatedly adding whichever predictor yields the best fit. As
more predictors are added, the p-values become larger. When this happens, if the
p-value for one of the variables exceeds a certain threshold, that variable is
removed from the model. The selection process continues in this forward and
backward manner until all the variables in the model have sufficiently low
p-values and all the predictors excluded from the model would result in a high
p-value if added to the model.

[glossary-backwards-selection]: glossary#backwards-selection "stats-learning-notes -- Glossary - Backwards Selection"
[glossary-coefficient]: glossary#coefficient "stats-learning-notes -- Glossary - Coefficient"
[glossary-confidence-interval]: glossary#confidence-interval "stats-learning-notes -- Glossary - Confidence Interval"
[glossary-correlation]: glossary#correlation "stats-learning-notes -- Glossary - Correlation"
[glossary-f-distribution]: glossary#f-distribution "stats-learning-notes -- Glossary - F-Distribution"
[glossary-f-statistic]: glossary#f-statistic "stats-learning-notes -- Glossary - F-Statistic"
[glossary-forward-selection]: glossary#forward-selection "stats-learning-notes -- Glossary - Forward Selection"
[glossary-hypothesis-test]: glossary#hypothesis-test "stats-learning-notes -- Glossary - Hypothesis Test"
[glossary-intercept]: glossary#intercept "stats-learning-notes -- Glossary - Intercept"
[glossary-least-squares-line]: glossary#least-squares-line "stats-learning-notes -- Glossary - Least Squares Line"
[glossary-mixed-selection]: glossary#mixed-selection "stats-learning-notes -- Glossary - Mixed Selection"
[glossary-multiple-linear-regression]: glossary#multiple-linear-regression "stats-learning-notes -- Glossary - Multiple Linear Regression"
[glossary-normal-distribution]: glossary#normal-distribution "stats-learning-notes -- Glossary - Normal Distribution"
[glossary-null-hypothesis]: glossary#null-hypothesis "stats-learning-notes -- Glossary - Null Hypothesis"
[glossary-null-model]: glossary#null-model "stats-learning-notes -- Glossary - Null Model"
[glossary-p-value]: glossary#p-value "stats-learning-notes -- Glossary - P-Value"
[glossary-parameter]: glossary#parameter "stats-learning-notes -- Glossary - Parameter"
[glossary-population-regression-line]: glossary#population-regression-line "stats-learning-notes -- Glossary - Population Regression Line"
[glossary-r-squared-statistic]: glossary#r-squared-statistic "stats-learning-notes -- Glossary - R-Squared Statistic"
[glossary-residual]: glossary#residual "stats-learning-notes -- Glossary - Residual"
[glossary-residual-standard-error]: glossary#residual-standard-error "stats-learning-notes -- Glossary - Residual Standard Error"
[glossary-residual-sum-of-squares]: glossary#residual-sum-of-squares "stats-learning-notes -- Glossary - Residual Sum of Sqaures"
[glossary-simple-linear-regression]: glossary#simple-linear-regression "stats-learning-notes -- Glossary - Simple Linear Regression"
[glossary-standard-error]: glossary#standard-error "stats-learning-notes -- Glossary - Standard Error"
[glossary-slope]: glossary#slope "stats-learning-notes -- Glossary - Slope"
[glossary-t-distribution]: glossary#t-distribution "stats-learning-notes -- Glossary - T-Distribution"
[glossary-t-statistic]: glossary#t-statistic "stats-learning-notes -- Glossary - T-Statistic"
[glossary-variable-selection]: glossary#variable-selection "stats-learning-notes -- Glossary - Varirable Selection"
[jekyll-site-chapter-3]: https://tdg5.github.io/stats-learning-notes/chapter-03.html "stats-learning-notes -- Chapter 3 - Linear Regression"

<a id="bottom"></a>
