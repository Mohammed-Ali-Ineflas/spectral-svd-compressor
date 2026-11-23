# Spectral Image Compressor via SVD (From Scratch)

A pure Python and NumPy implementation of Singular Value Decomposition (SVD) for low-rank image approximation and dimensionality reduction. Built without high-level linear algebra wrappers (`np.linalg.svd` or `scikit-learn`) to demonstrate the underlying matrix factorization mechanics.

## Mathematical Formulation

Given a grayscale image represented as a real matrix $A \in \mathbb{R}^{m \times n}$, SVD factors $A$ into:

$$A = U \Sigma V^T$$

1. **Gram Matrix & Eigendecomposition:** Compute the symmetric matrix $M = A^T A \in \mathbb{R}^{n \times n}$ and extract its eigenvalues $\lambda_i$ and eigenvectors $v_i$ (`np.linalg.eigh`).
2. **Singular Values:** Sort eigenvalues in descending order and compute $\sigma_i = \sqrt{\lambda_i}$.
3. **Left Singular Vectors:** Project $A$ onto the right singular vectors and normalize: $u_i = \frac{1}{\sigma_i} A v_i$.
4. **Low-Rank Approximation:** Truncate the matrices to the top $k$ components ($A_k = U_{:, :k} \Sigma_{:k, :k} V_{:k, :}^T$) to minimize reconstruction error under the Frobenius norm.

## Setup & Usage

1. Clone the repository and set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt