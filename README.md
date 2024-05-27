# NFT Infra index Methodology
## Econometric estimation of an IRL-based market portfolio model

In this notebook we: 

- Explore and estimate an IRL-based model of market returns that is based on IRL of a market-optimal portfolio 
- Investigate the role and impact of choices of different signals on model estimation and trading strategies
- Compare simple IRL-based and UL-based trading strategies

**Pipeline structure:**

- **Part 1**: Complete the model estimation for the DJI portfolio of 30 stocks, and simple signals such as simple moving averages constructed below.

- **Part 2**: Propose other signals and investigate the dynamics for market caps obtained with alternative signals. Present your conclusions and observations.

- **Part 3**: Can you repeat your analysis for the S&P portfolio? You will have to build a data file, build signals, and repeat the model estimation process with your new dataset.

- **Part 4** : We will build a strategy using an optimal market-implied policy estimated from this model, and compare it with PCA and absorption ratio strategies.

**Instructions on packages**

- We will be using Python 3 in this project.
- We use TensorFlow Compat V1


**Objective:**
- Get experience with building and estimation of an IRL based model of market dynamics, and learn how this IRL approach extends the famous Black-Litterman model (see F. Black and R. Litterman, "Global Portfolio Optimization", Financial Analyst Journal, Sept-Oct. 1992, 28-43, and  D. Bertsimas, V. Gupta, and I.Ch. Paschalidis, "Inverse Optimization: A New Perspective on the Black-Litterman Model", Operations Research, Vol.60, No.6, pp. 1389-1403 (2012), I.Halperin and I. Feldshteyn "Market Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand Inference with Free Energy", https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3174498). 
- Know how to enhance a market-optimal portfolio policy by using your private signals. 
- Be able to implement trading strategies based on this method.

## The IRL-based model

The optimal investment policy in the problem of inverse portfolio optimization is a Gaussian policy (see article):

$$ \pi_{\theta}({\bf a}_t |{\bf y}_t ) =   \mathcal{N}\left({\bf a}_t | \bf{A}_0 + \bf{A}_1 {\bf y}_t, \Sigma_p \right) $$

Here ${\bf y}_t$ is a vector of dollar position in the portfolio, and $\bf{A}_0$, $\bf{A}_1$ and $\Sigma_p$ are parameters defining a Gaussian policy.   

Such Gaussian policy is found for both cases of a single investor and a market portfolio. It's computed through a numerical scheme that can iteratively compute coefficients $\bf{A}_0$, $\bf{A}_1$ and $\Sigma_p$ using a combination of a RL algorithm called G-learning and a trajectory optimization algorithm.

In this notebook, we will explore implications and estimation of this IRL-based model for the most interesting case - the market portfolio. It turns out that for this case, the model can be estimated in an easier way using a conventional Maximum Likelihood approach. To this end, we re-formulate the model for this particular case in three easy steps.


Recall that for a vector of $N$ stocks, we introduced a size $2 N$-action vector $`{\bf a}_t=[{\bf u}_t^{(+)}, {\bf u}_t^{(-)}]`$, so that an action ${\bf u}_t$ was defined as a difference of two non-negative numbers $`{\bf u}_t = {\bf u}_t^{(+)} - {\bf u}_t^{(-)} = [{\bf 1}, - {\bf 1}] {\bf a}_t \equiv {\bf 1}{-1}^{T} {\bf a}_t`$.

Therefore, the joint distribution of $`{\bf a}_t=[{\bf u}_t^{(+)}, {\bf u}_t^{(-)}]`$ is given by our Gaussian policy $`\pi_{\theta}(\bf{a}_t|\bf{y}_t)`$. This means that the distribution of ${\bf u}_t = {\bf u}_t^{(+)} - {\bf u}_t^{(-)}$ is also Gaussian. Let us write it therefore as follows: 

$$
\pi_{\theta}({\bf u}_t |{\bf y}_t ) =   \mathcal{N}\left({\bf u}_t | \bf{U}_0 + \bf{U}_1 {\bf y}_t, \Sigma_u \right) 
$$

Here $`\bf{U}_{0}={\bf 1}_{-1}^{T}\bf{A}_0`$ and $`\bf{U}_1={\bf 1}_{-1}^{T}\bf{A}_1`$.

This means that ${\bf u}_t$ is a Gaussian random variable that we can write as follows:

$$
{\bf u}_t = \bf{U}_0 + \bf{U}_1 {\bf y}_t + \varepsilon_t^{(u)}  = \bf{U}_0 + \bf{U}_1^{(x)} {\bf x}_t + \bf{U}_1^{(z)} {\bf z}_t + \varepsilon_t^{(u)} 
$$

where $\varepsilon_t^{(u)} \sim \mathcal{N}(0,\Sigma_u)$ is a Gaussian random noise.  

The most important feature of this expression that we need going forward is is linear dependence on the state ${\bf x}_t$. 
This is the only result that we will use in order to construct a simple dynamic market model resulting from our IRL model. We use a deterministic limit of this equation, where in addition we set $\bf{U}_0 = \bf{U}_1^{(z)} = 0$, and replace $\bf{U}_1^{(x)} \rightarrow \phi$ to simplify the notation. We thus obtain a simple deterministic policy:

$$
{\bf u}_t =  \phi  {\bf x}_t 
$$

Next, let us recall the state equation and return equation (where we reinstate a time step $\Delta t$,
and $\circ$ stands for an element-wise (Hadamard) product):

$$
X_{t+ \Delta t} = (1 + r_t \Delta t) \circ (  X_t +  u_t  \Delta t)  
$$

$$
r_t=r_f+{\bf w} {\bf z}_t -\mu  u_t + \frac{\sigma}{\sqrt{\Delta t}} \varepsilon_t 
$$

where $r_f$ is a risk-free rate, $\Delta t$ is a time step, ${\bf z}_t$ is a vector of predictors with weights ${\bf w}$, $\mu$ is a market impact parameter with a linear impact specification, and $\varepsilon_t \sim \mathcal{N} (\cdot| 0, 1)$ is a white noise residual.


Eliminating $u_t$ from these expressions and simplifying, we obtain

$$ \Delta  X_t = \mu  \phi  ( 1 + \phi \Delta t) \circ  X_t \circ \left(  \frac{r_f (1 + \phi \Delta t)  + \phi}{ \mu \phi (1+ \phi \Delta t )}  -  X_t \right) \Delta t + 
( 1 + \phi \Delta t) X_t  \circ \left[ {\bf w} {\bf z}_t  \Delta t +  \sigma \sqrt{ \Delta t} \varepsilon_t \right]
$$

Finally, assuming that $\phi \Delta t \ll 1$ and taking the continuous-time limit $\Delta t \rightarrow dt$, we obtain 

$$
d X_t = \kappa \circ X_t \circ \left( \frac{\theta}{\kappa} - X_t \right) dt +  X_t \circ \left[{\bf w} {\bf z}_t \, dt + \sigma d W_t \right]
$$

where $`\kappa =\mu\phi`$, $`\theta=r_f +\phi`$, and $`W_t`$ is a standard Brownian motion.

Please note that this equation describes dynamics with a quadratic mean reversion. It is quite different from models with linear mean reversion such as the Ornstein-Uhlenbeck (OU) process. 

Without signals ${\bf z}_t$, this process is known in the literature as a Geometric Mean Reversion (GMR) process. It has been used (for a one-dimensional setting) by Dixit and Pyndick (" Investment Under Uncertainty", Princeton 1994), and investigated (also for 1D) by Ewald and Yang ("Geometric Mean Reversion: Formulas for the Equilibrium Density and Analytic Moment Matching", University of St. Andrews Economics Preprints, 2007). We have found that such dynamics (in a multi-variate setting) can also be obtained for market caps (or equivalently for stock prices, so long as the number of shares is held fixed) using Inverse Reinforcement Learning! 

(For more details, see I. Halperin and I. Feldshteyn, "Market Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand Inference with Free Energy.
(or, How We Learned to Stop Worrying and Love Bounded Rationality)", https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3174498)

## Model calibration with moving average signals 
Recall the equation for the dynamics of market portfolio: 

$$ \Delta {\bf x}_t = \kappa_x \circ  {\bf x}_t \circ \left( {\bf W}{\bf z}_t'  - {\bf x}_t \right)  +  {\bf x}_t  \circ \varepsilon_t^{(x)} $$

Here we change the notation a bit. Now ${\bf z}_t'$ is an extended vector of predictors that includes a constant unit predictor $`{\bf z}_t' = [1, {\bf z}_t ]^T`$. Therefore, for each name, if you have $K = 2$ signals, an extended vector of signals ${\bf z}_t'$ is of length $K + 1$, and the  $W$ stands for a factor loading matrix.
The negative log-likelihood function for observable data with this model is therefore

$$ LL_M(\Theta) = - \log\prod_{t=0}^{T-1} \frac{1}{\sqrt{(2 \pi)^{N} \left| \Sigma_x \right| }} e^{ - \frac{1}{2} \left({\bf v}_t \right)^{T} \Sigma_x^{-1}\left({\bf v}_t \right)} $$

where

$$ 
{\bf v}_t \equiv \frac{{\bf x}_{t+1}-{\bf x}_t} {{\bf x}_t} - \kappa_x \circ \left({\bf W} {\bf z}_t' - {\bf x}_t \right)
$$

and $\Sigma_x$ is the covariance matrix that was specified above in terms of other parameters. Here we directly infer the value of $\Sigma_x$, along with other parameters, from data, so we will not use these previous expressions. 

Parameters that you have to estimate from data are therefore the vector of mean reversion speed 
parameters $\kappa_x$, factor loading matrix ${\bf W} \equiv {\bf w}_z'$, and covariance matrix $\Sigma_x$. 

Now, you are free to impose some structure on this parameters. Here are some choice, in the order of increasing complexity:

- Assume that all values in vector-valued and matrix-valued parameters are the same, so that they can parametrized by scalars, e.g. $\kappa_x = \kappa {\bf 1}_N$ where $\kappa$ is a scalar value, and ${\bf 1}_N$ is a vector of ones of length $N$ where $N$ is the number of stocks in the market portfolio. You can proceed similarly with specification of factor loading matrix $W'$. Assume that all values in (diagonal!) factor loading matrices are the same for all names, and assume that all correlations and variances in the covariance matrix $\Sigma_x$ are the same for all names.   

- Assume that all values are the same only within a given industrial sector.

- You can also change the units. For example, you can consider logs of market caps instead of market caps themselves, ie. change the variable from ${\bf x}_t$ to ${\bf q}_t = \log {\bf x}_t$

