---
layout: default
use_math: true
---

## [Chapter 3 - Linear Regression][jekyll-site-chapter-3]

[Simple linear regression][glossary-simple-linear-regression] predicts a
quantitative response $$ Y $$ on the basis of a single predictor variable $$ X
$$. It assumes an approximately linear relationship between $$ X $$ and $$ Y $$.
Formally,

$$ Y \approx \beta_{0} + \beta_{1}X $$

where $$ \beta_{0} $$ represents the [intercept][glossary-intercept] or the
value of $$ Y $$ when $$ X $$ is equal to 0 and $$ \beta_{1} $$ represents the
[slope][glossary-slope] of the line or the average amount of change in $$ Y $$
for each one-unit increase in $$ X $$.

Together, $$ \beta_{0} $$ and $$ \beta_{1} $$ are known as the model
[coefficients][glossary-coefficient] or [parameters][glossary-parameter].

### Estimating Model Coefficients

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

$$ RSS = e_{1}^2 + e_{2}^2 + ... + e_{n}^2 $$

or

$$ RSS = (y_{1} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{1})^2 +
(y_{2} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{2})^2 +. ... +
(y_{n} - \hat{\beta_{0}} - \hat{\beta_{1}}x_{n})^2 .$$

Assuming sample means of

$$ \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_{i} $$

and

$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_{i} , $$

calculus can be applied to estimate the least squares coefficient estimates for
linear regression to minimize the residual sum of squares like so

$$ \beta_{1} = \frac{\sum_{i=1}^{n}(x_{i} - \bar{x})(y_{i} -
\bar{y})}{\sum_{i=1}^{n}(x_{i} - \bar{x})^2} $$

$$ \beta_{0} = \bar{y} - \hat{\beta_{1}}\bar{x} $$

### Assessing Coefficient Estimate Accuracy

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

[glossary-coefficient]: glossary#coefficient "stats-learning-notes -- Glossary - Coefficient"
[glossary-intercept]: glossary#intercept "stats-learning-notes -- Glossary - Intercept"
[glossary-least-squares-line]: glossary#least-squares-line "stats-learning-notes -- Glossary - Least Squares Line"
[glossary-parameter]: glossary#parameter "stats-learning-notes -- Glossary - Parameter"
[glossary-population-regression-line]: glossary#population-regression-line "stats-learning-notes -- Glossary - Population Regression Line"
[glossary-residual]: glossary#residual "stats-learning-notes -- Glossary - Residual"
[glossary-residual-sum-of-squares]: glossary#residual-sum-of-squares "stats-learning-notes -- Glossary - Residual Sum of Sqaures"
[glossary-simple-linear-regression]: glossary#simple-linear-regression "stats-learning-notes -- Glossary - Simple Linear Regression"
[glossary-slope]: glossary#slope "stats-learning-notes -- Glossary - Slope"
[jekyll-site-chapter-3]: https://tdg5.github.io/stats-learning-notes/chapter-03.html "stats-learning-notes -- Chapter 3 - Linear Regression"

<a id="bottom"></a>
