import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_grayscale_image(image_path: str) -> np.ndarray:
    """Load an image from disk, convert to grayscale, and normalize pixel values to [0, 1]."""
    img = Image.open(image_path).convert('L')
    return np.asarray(img, dtype=np.float64) / 255.0


def compute_svd_scratch(A: np.ndarray):
    """
    Compute the Singular Value Decomposition (A = U @ diag(S) @ Vt) 
    from scratch using the eigendecomposition of the Gram matrix (A^T @ A).
    """
    # Compute the symmetric positive semi-definite matrix A^T * A
    ATA = A.T @ A
    
    # Extract eigenvalues and right singular vectors (V)
    eigenvalues, V = np.linalg.eigh(ATA)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[sorted_idx], 0.0)  # clamp tiny negative numerical noise
    V = V[:, sorted_idx]

    # Singular values are the square roots of the eigenvalues
    singular_values = np.sqrt(eigenvalues)

    # Filter out near-zero singular values to prevent division by zero when computing U
    valid_mask = singular_values > 1e-10
    S = singular_values[valid_mask]
    V = V[:, valid_mask]

    # Derive left singular vectors: u_i = (A @ v_i) / sigma_i
    U = (A @ V) / S

    return U, S, V.T


def reconstruct_rank_k(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct a low-rank approximation of the original matrix using the top k components."""
    k = min(k, len(S))
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    return U_k @ S_k @ Vt_k


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray, k: int):
    """Calculate the relative Frobenius reconstruction error and storage reduction ratio."""
    fro_error = np.linalg.norm(original - reconstructed, 'fro') / np.linalg.norm(original, 'fro')
    
    m, n = original.shape
    raw_size = m * n
    compressed_size = k * (m + n + 1)
    space_saved = 1.0 - (compressed_size / raw_size)
    
    return fro_error, space_saved


def plot_compression_results(original: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, ranks=(5, 20, 50, 100)):
    """Visualize the original image alongside its rank-k approximations."""
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(16, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"Original\n({original.shape[0]}x{original.shape[1]})")
    axes[0].axis('off')
    
    for idx, k in enumerate(ranks):
        rec = reconstruct_rank_k(U, S, Vt, k)
        rec = np.clip(rec, 0.0, 1.0)
        err, saved = compute_metrics(original, rec, k)
        
        axes[idx + 1].imshow(rec, cmap='gray')
        axes[idx + 1].set_title(f"Rank k={k}\nSaved: {saved*100:.1f}%\nError: {err*100:.1f}%")
        axes[idx + 1].axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Numerical verification on a deterministic 3x3 matrix
    toy_matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)

    U_toy, S_toy, Vt_toy = compute_svd_scratch(toy_matrix)
    rec_toy = reconstruct_rank_k(U_toy, S_toy, Vt_toy, k=len(S_toy))
    print(f"Toy Matrix Max Reconstruction Error: {np.max(np.abs(toy_matrix - rec_toy)):.2e}")

    # 2. Image compression test (uncomment once you add 'sample.jpg' to the folder)
    # img = load_grayscale_image("sample.jpg")
    # U, S, Vt = compute_svd_scratch(img)
    # plot_compression_results(img, U, S, Vt)