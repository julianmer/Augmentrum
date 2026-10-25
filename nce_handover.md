# Handover Context: Noncentral Chi Error (NCE) Implementation

## 📌 Objective
Implement a custom evaluation metric called **Noncentral Chi Error (NCE)** to address the "hidden noise problem" in MR image reconstruction. This metric evaluates reconstruction quality against a noisy reference image without penalizing the model for failing to reconstruct the underlying, unobserved thermal noise.

---

## 🔬 Mathematical Foundation

### 1. The Core NCE Metric
Instead of standard MSE/RMSE (which incorrectly treats a noisy reference as absolute ground truth), NCE calculates a normalized version of the classical **negative log-likelihood** under a Noncentral Chi (NCC) noise distribution:

$$\text{NCE}(x, y) = -\ln p(y \mid x) + \ln p(y \mid y)$$

*   $x$: Reconstructed image array (predicted).
*   $y$: Noisy reference image array (target).
*   $-\ln p(y \mid x)$: Negative log-likelihood measuring data consistency.
*   $\ln p(y \mid y)$: Normalization constant ensuring $\text{NCE}(y, y) = 0$.

### 2. The Statistical Noise Model
For magnitude MR images reconstructed via root-sum-of-squares (rSoS) from $L$ receiver coils, the intensity $y_n$ at voxel $n$ follows a Noncentral Chi distribution:

$$p(y_n \mid x_n) = \frac{x_n^{1-L}}{\sigma^2} y_n^L \exp\left(-\frac{y_n^2 + x_n^2}{2\sigma^2}\right) I_{L-1}\left(\frac{x_n y_n}{\sigma^2}\right)$$

Where:
*   $\sigma$: Noise standard deviation parameter.
*   $L$: Effective degrees of freedom (number of coils).
*   $I_{L-1}$: Modified Bessel function of the first kind of order $L-1$.

---

## 🛠️ Algorithmic Implementation Steps

Assuming pixel independence, the total log-likelihood is the sum across all $N$ voxels. 

### Step 1: Compute $-\ln p(y \mid x)$
$$\sum_{n=1}^{N} \left[ (L-1)\ln\left(\frac{x_n}{y_n}\right) + \frac{y_n^2 + x_n^2}{2\sigma^2} - \ln I_{L-1}\left(\frac{x_n y_n}{\sigma^2}\right) + 2\ln\sigma \right]$$

### Step 2: Compute $\ln p(y \mid y)$
$$\sum_{n=1}^{N} \left[ -\frac{y_n^2}{\sigma^2} + \ln I_{L-1}\left(\frac{y_n^2}{\sigma^2}\right) - 2\ln\sigma \right]$$

### Step 3: Combine
$$\text{NCE} = [-\ln p(y \mid x)] + [\ln p(y \mid y)]$$

*Note: In the final subtraction, constant terms like $2\ln\sigma$ cancel out, but keeping them explicitly in code blocks helps with unit testing individual likelihood functions.*

---

## 💻 Claude Code Implementation Tasks

1. **Numerical Stability**: The modified Bessel function $I_{L-1}(z)$ grows exponentially. You **must** use exponentially scaled variants (like `scipy.special.ive` or log-Bessel approximations) to prevent `NaN` or `Inf` errors when handling high SNR voxels where $z = \frac{x y}{\sigma^2}$ is large.
2. **Handle Zeros**: Protect against division by zero or $\ln(0)$ if background voxels contain absolute zeros. Add a small epsilon ($\epsilon = 1e-9$) where necessary.
3. **Array Operations**: Ensure the code supports element-wise matrix/tensor calculations across 2D or 3D image volumes.
