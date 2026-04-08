---
layout: default
use_math: true
---

[Previous: Chapter 3 - Linear Regression][chapter-03-linear-regression]

---

## [Chapter 4 - Classification][chapter-04-classification]

Though linear regression can be applied in the case of binary qualitative
responses, difficulties arise beyond two levels. For example, choosing a coding
scheme is problematic, and different coding schemes can yield very different
predictions.

### Logistic Regression

[Logistic regression][glossary-logistic-regression] models the probability that
$$ y $$ belongs to a particular category rather than modeling the response
itself.

Logistic regression uses the [logistic function][glossary-logistic-function] to
ensure predictions lie between $$ 0 $$ and $$ 1 $$. The logistic function takes
the form

$$ \normalsize p(X) = \frac{e^{\beta_{0} + \beta_{1}X}}{1 + e^{\beta_{0} +
\beta_{1}X}} .$$

This yields a probability greater than $$ 0 $$ and less than $$ 1 $$.

The logistic function can be rearranged to yield

$$ \normalsize \frac{p(X)}{1 - p(X)} = e^{\beta_{0} + \beta_{1}X} $$

$$ \frac{p(X)}{1 - p(X)} $$ is known as the [odds][glossary-odds] and takes on
values between $$ 0 $$ and infinity.

As an example, a probability of 1 in 5 yields odds of $$ \frac{1}{4} $$ since
$$ \frac{0.2}{1 - 0.2} = \frac{1}{4} .$$

Taking a logarithm of both sides yields the [log-odds][glossary-log-odds] (or
[logit][glossary-logit]):

$$ \normalsize \log \bigg( \frac{p(X)}{1 - p(X)} \bigg) =
\beta_{0} + \beta_{1}X $$

Logistic regression has a logit that is linear in terms of $$ X $$.

Unlike linear regression, where $$ \beta_{1} $$ represents the average change in
$$ Y $$ for a one-unit increase in $$ X $$, in logistic regression, increasing
$$ X $$ by one unit results in a $$ \beta_{1} $$ change in the log-odds. This is
equivalent to multiplying the odds by $$ e^{\beta_{1}} $$.

The relationship between $$ p(X) $$ and $$ X $$ is not linear. Thus,
$$ \beta_{1} $$ does not correspond directly to the change in $$ p(X) $$ for a
one-unit increase in $$ X $$. However:
- If $$ \beta_{1} > 0 $$, increasing $$ X $$ increases $$ p(X) $$
- If $$ \beta_{1} < 0 $$, increasing $$ X $$ decreases $$ p(X) $$

---

#### Estimating Regression Coefficients

Logistic regression uses [maximum likelihood][glossary-maximum-likelihood] to
estimate coefficients.

We choose $$ \beta_{0} $$ and $$ \beta_{1} $$ such that predicted probabilities
$$ \hat{p}(x_i) $$ align closely with observed classes.

This yields the [likelihood function][glossary-likelihood-function]:

$$ \normalsize \ell(\beta_{0}, \beta_{1}) =
\prod_{i:y_{i}=1} p(X_{i}) \times
\prod_{j:y_{j}=0} (1 - p(X_{j})) .$$

We select parameters that maximize this function.

Linear regression's least squares is a special case of maximum likelihood.

Logistic regression evaluates coefficient accuracy using the
[z-statistic][glossary-z-statistic]:

$$ \normalsize \textrm{z-statistic}(\beta_{1}) =
\frac{\hat{\beta}_{1}}{\mathrm{SE}(\hat{\beta}_{1})} $$

A large z-statistic provides evidence against the null hypothesis.

The null hypothesis is:

$$ \normalsize H_{0}: \beta_{1} = 0 $$

which implies:

$$ \normalsize p(X) =
\frac{e^{\beta_{0}}}{1 + e^{\beta_{0}}} $$

Thus, $$ p(X) $$ does not depend on $$ X $$.

---

#### Making Predictions

Predictions are made using:

$$ \normalsize \hat{p}(X) =
\frac{e^{\hat{\beta}_{0} + \hat{\beta}_{1}X}}
{1 + e^{\hat{\beta}_{0} + \hat{\beta}_{1}X}} .$$

The intercept $$ \hat{\beta}_{0} $$ is often less interpretable, as it mainly
captures the baseline class balance.

[Dummy variables][glossary-dummy-variable] can be used for qualitative predictors.

---

#### Multiple Logistic Regression

Generalization:

$$ \normalsize \log \bigg( \frac{p(X)}{1 - p(X)} \bigg) =
\beta_{0} + \beta_{1}X_{1} + \cdots + \beta_{p}X_{p} $$

Equivalent probability form:

$$ \normalsize p(X) =
\frac{e^{\beta_{0} + \beta_{1}X_{1} + \cdots + \beta_{p}X_{p}}}
{1 + e^{\beta_{0} + \beta_{1}X_{1} + \cdots + \beta_{p}X_{p}}} $$

---

Confounding occurs when relationships between predictors distort results.

---

### Linear Discriminant Analysis

[Linear discriminant analysis][glossary-linear-discriminant-analysis] models
$$ X $$ given $$ Y $$ and applies
[Bayes' theorem][glossary-bayes-theorem].

---

#### Bayes' Theorem

$$ \normalsize \mathrm{Pr}(Y=k|X=x) = p_{k}(x) =
\frac{\pi_{k}f_{k}(x)}{\sum_{j=1}^{K}\pi_{j}f_{j}(x)} $$

- $$ \pi_k $$ = prior probability  
- $$ f_k(x) $$ = class density  
- $$ p_k(x) $$ = posterior probability  

---

#### LDA (One Predictor)

Assume normal distribution:

$$ \normalsize f_{k}(x) =
\frac{1}{\sqrt{2\pi}\sigma}
\exp \left(-\frac{(x-\mu_{k})^{2}}{2\sigma^{2}} \right) $$

Discriminant function:

$$ \normalsize \delta_{k}(x) =
\frac{x\mu_{k}}{\sigma^{2}} -
\frac{\mu_{k}^{2}}{2\sigma^{2}} +
\log(\pi_{k}) $$

---

Estimates:

$$ \normalsize \hat{\mu}_{k} =
\frac{1}{n_{k}} \sum_{i:y_{i}=k} x_{i} $$

$$ \normalsize \hat{\sigma}^{2} =
\frac{1}{n - K} \sum_{k=1}^{K}
\sum_{i:y_{i}=k} (x_{i} - \hat{\mu}_{k})^{2} $$

$$ \normalsize \hat{\pi}_{k} = \frac{n_{k}}{n} $$

---

#### Multivariate LDA

$$ \normalsize X \sim N(\mu, \Sigma) $$

$$ \normalsize \delta_{k}(x) =
x^{T}\Sigma^{-1}\mu_{k} -
\frac{1}{2}\mu_{k}^{T}\Sigma^{-1}\mu_{k} +
\log \pi_{k} $$

---

### Quadratic Discriminant Analysis

[Quadratic discriminant analysis][glossary-quadratic-discriminant-analysis]
assumes class-specific covariance:

$$ \normalsize X \sim N(\mu_{k}, \Sigma_{k}) $$

$$ \normalsize \delta_{k}(x) =
- \frac{1}{2}(x - \mu_{k})^{T}\Sigma_{k}^{-1}(x - \mu_{k})
- \frac{1}{2}\log|\Sigma_{k}| +
\log \pi_{k} $$

---

### Comparing Methods

- Logistic Regression → simple, interpretable  
- LDA → better when Gaussian assumption holds  
- QDA → more flexible, higher variance  
- KNN → non-linear, less interpretable  

---

[Next: Chapter 5 - Resampling Methods][chapter-05-resampling-methods]

... (links unchanged)
