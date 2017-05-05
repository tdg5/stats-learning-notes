---
layout: default
use_math: true
---

## Chapter 2 - Statistical Learning

[Inputs][glossary-input], also known as predictors, independent variables, features, or more
generally, variables.

[Outputs][glossary-output], also known as response or dependent variable.

Suppose an observed quantitative response $$ {Y} $$ and $$ {p} $$ different
predictors $$ x_{1}, x_{2}, \ldots, x_{p} $$. The assumed relationship between $$
{Y} $$ and $$ X = (x_{1}, x_{2}, \ldots, x_{p}) $$ can be generalized as:

$$ {Y} = f(X) + \epsilon $$

where $$ {f} $$ is some fixed, but unknown function of $$ {X} $$ and $$ \epsilon
$$ is a random [error term][glossary-irreducible-error] that is independent of
$$ {X} $$ and has a mean of zero. In such a scenario, $$ f $$ represents the
systematic information that $$ X $$ provides about $$ Y $$.

In general, an estimation of $$ f $$, denoted by $$ \hat{f} $$, will not be
perfect and will introduce error.

The error introduced by the discrepancy between $$ f $$ and $$ \hat{f} $$ is
known as [irreducible error][glossary-irreducible-error] because it can never be
reduced regardless of the accuracy $$ \hat{f} $$.

The irreducible error will be larger than zero because $$ \epsilon $$ may
contain unmeasured variables needed to predict $$ {Y} $$ or $$ \epsilon $$ may
contain unmeasured variation. The irreducible error always enforces an upper
bound on the accuracy of predicting $$ {Y} $$. In practice, this bound is almost
always unknown.

### Estimating $$ {f} $$

#### Model Interpretability vs. Prediction Accuracy Trade-Off

##### Parametric Methods

[Parametric methods][glossary-parametric-methods] utilize a two-step model-based
approach.

1. First, make an assumption about the functional nature, or shape, of $$ {f}
$$. For example, assume that $$ {f} $$ is linear, yielding a linear model.
2. Once a model has been selected, use training data to fit, or train, the
model. In the case of a linear model of the form

$$ f(x) = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots + \beta_{p}x_{p}, $$

the training procedure should yield estimates for the parameters $$ \beta_{0},
\beta_{1}, \beta_{2}, \ldots, \beta_{p} $$ such that

$$ {Y} \approx f({X}) \approx \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots + \beta_{p}x_{p}. $$

A model-based approach like that outlined above is referred to as
[parametric][glossary-parametric] because it simplifies the problem of
estimating $$ {f} $$ down to estimating a set of parameters.

In general, it is much simpler to estimate a set of parameters than it is to
estimate an entirely arbitrary function $$ {f} $$. A disadvantage of this
approach is that the specified model won't usually match the true form of $$ f
$$.

Using more flexible models is one means to attempt to combat inaccuracies in the
chosen mode. However, more flexible models have the disadvantage of requiring a
greater number of parameters to be estimated and they are also more susceptible
to overfitting.

[Overfitting][glossary-overfitting] is a phenomenon where a model closely
matches the training data such that it captures too much of the noise or error
in the data. This results in a model that fits the training data very well, but
doesn't make good predictions under test or in general.

##### Non-Parametric Methods

[Non-parametric methods][glossary-non-parametric-methods] don't make explicit
assumptions about $$ f $$ and instead seek to estimate $$ f $$ by getting as
close to the data points as possible without being too coarse or granular,
preferring smoothness instead.

[Non-parametric][glossary-non-parametric-methods] approaches can fit a wider
range of possible shapes for $$ {f} $$ since essentially no assumptions about
the form of $$ {f} $$ are made. However, since non-parametric approaches don't
simplify the problem of estimating $$ {f} $$, they tend to require a very large
number of observations to accurately estimate $$ {f} $$.

A thin-plate spline is one example of a non-parametric method.

Though less flexible, more restrictive models are more limited in the shapes
they can estimate, they are easier to interpret because the relation of the
predictors to the output is more easily understood.

#### Supervised Learning vs. Unsupervised Learning

[Supervised learning][glossary-supervised-learning] refers to those scenarios in
which for each observation of the predictor measurements $$ X_{i} $$ there is an
associated response measurement $$ Y_{i} $$. In such a scenario, it is often
desirable to generate a model that relates the predictors to the response with
the goal of accurately predicting future observations or of better inferring the
relationship between the predictors and the response.

[Unsupervised learning][glossary-unsupervised-learning] refers to those
scenarios in which for each observation of the predictor measurements $$ X_{i}
$$, there is no associated response $$ Y_{i} $$. This is referred to as
unsupervised because there is no response variable that can supervise the
analysis that goes into generating a model.

[Cluster analysis][glossary-cluster-analysis], a process by which observations
are arranged into relatively distinct groups, is one form of unsupervised
learning.

#### Regression Problems vs. Classification Problems

[Quantitative values][glossary-quantitative-value], whether a variable or
response, take on numerical values. Problems with a quantitative response are
often referred to as [regression problems][glossary-regression-problem].

[Qualitative values][glossary-qualitative-value] whether a variable or response,
take on values in one of $$ K $$ different class or categories. Problems with a
qualitative response are often referred to as [classification
problems][glossary-classification-problem].

Which statistical learning method is best suited to a problem tends to depend on
whether the response is qualitative or quantitative.

### Measuring Quality of Fit

[glossary-classification-problem]: glossary#classification-problem "stats-learning-notes \| Glossary - Classification Problem"
[glossary-cluster-analysis]: glossary#cluster-analysis "stats-learning-notes \| Glossary - Cluster Analysis"
[glossary-error-term]: glossary#error-term "stats-learning-notes \| Glossary - Error Term"
[glossary-input]: glossary#input "stats-learning-notes \| Glossary - Input"
[glossary-irreducible-error]: glossary#irreducible-error "stats-learning-notes \| Glossary - Irreduicible Error"
[glossary-non-parametric]: glossary#non-parametic "stats-learning-notes \| Glossary - Non-Parametric"
[glossary-non-parametric-methods]: glossary#non-parametic-methods "stats-learning-notes \| Glossary - Non-Parametric methods"
[glossary-output]: glossary#output "stats-learning-notes \| Glossary - Output"
[glossary-overfitting]: glossary#overfitting "stats-learning-notes \| Glossary - Overfitting"
[glossary-parametric-methods]: glossary#parametic-methods "stats-learning-notes \| Glossary - Parametric methods"
[glossary-parametric]: glossary#parametic "stats-learning-notes \| Glossary - Parametric"
[glossary-regression-problem]: glossary#regression-problem "stats-learning-notes \| Glossary - Regression Problem"
[glossary-qualitative-value]: glossary#qualitative-value "stats-learning-notes \| Glossary - Qualitative Value"
[glossary-quantitative-value]: glossary#quantitative-value "stats-learning-notes \| Glossary - Quantitative Value"
[glossary-supervised-learning]: glossary#supervised-learning "stats-learning-notes \| Glossary - Supervised Learning"
[glossary-unsupervised-learning]: glossary#unsupervised-learning "stats-learning-notes \| Glossary - Unsupervised Learning"
