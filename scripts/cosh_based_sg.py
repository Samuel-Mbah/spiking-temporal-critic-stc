
import numpy as np
import matplotlib.pyplot as plt
import argparse
    
    # Fix matplotlib backend issue - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend


# Function to plot and compare surrogate gradients
def plot_surrogate_comparison(save_path="Plots/surrogate_comparison.png", 
                               alpha1=10.0, beta1=1.0, alpha2=5.0, beta2=1.0):
    """
    Plot comparison between fast sigmoid and Wang et al. cosh surrogates
    
    Args:
        alpha1, beta1: First cosh surrogate parameters
        alpha2, beta2: Second cosh surrogate parameters
    """
    
    
    # Define membrane potential range around threshold (m - θ)
    m_minus_theta = np.linspace(-2.0, 2.0, 1000)
    
    # Fast sigmoid surrogate (snntorch default)
    beta_sigmoid = 25.0  # typical value
    fast_sigmoid = 1.0 / (1.0 + np.exp(-beta_sigmoid * m_minus_theta))
    fast_sigmoid_grad = beta_sigmoid * fast_sigmoid * (1 - fast_sigmoid)
    
    # Wang et al. cosh surrogate with different parameters
    # Use the passed parameters instead of hardcoded values
    
    cosh_grad1 = 1.0 / (np.cosh(alpha1 * m_minus_theta) ** beta1)
    cosh_grad2 = 1.0 / (np.cosh(alpha2 * m_minus_theta) ** beta2)
    
    plt.figure(figsize=(12, 5))
    
    # Plot surrogate gradients
    plt.subplot(1, 2, 1)
    plt.plot(m_minus_theta, fast_sigmoid_grad, 'b-', label='Fast Sigmoid (β=25)', linewidth=2)
    plt.plot(m_minus_theta, cosh_grad1, 'r-', label=f'Cosh (α={alpha1}, β={beta1})', linewidth=2)
    plt.plot(m_minus_theta, cosh_grad2, 'g-', label=f'Cosh (α={alpha2}, β={beta2})', linewidth=2)
    plt.xlabel('Membrane Potential - Threshold (m - θ)')
    plt.ylabel('Surrogate Gradient')
    plt.title('Surrogate Gradient Functions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot forward pass (spike functions)  
    plt.subplot(1, 2, 2)
    heaviside = (m_minus_theta > 0).astype(float)
    
    # Create soft spike approximations that match the cosh gradient characteristics
    # These show how the different surrogate gradients would shape a differentiable spike
    
    # Method 1: Use tanh-based approximation that relates to cosh gradient
    # Since d/dx[tanh(αx)] ∝ sech²(αx) = 1/cosh²(αx), we can relate them
    cosh_soft1 = 0.5 * (1 + np.tanh(alpha1 * m_minus_theta / 2))  # gentler slope
    cosh_soft2 = 0.5 * (1 + np.tanh(alpha2 * m_minus_theta / 2))  # even gentler
    
    plt.plot(m_minus_theta, heaviside, 'k-', label='True Spike (Heaviside)', linewidth=3)
    plt.plot(m_minus_theta, fast_sigmoid, 'b--', label='Fast Sigmoid (β=25)', linewidth=2, alpha=0.8)
    plt.plot(m_minus_theta, cosh_soft1, 'r:', label=f'Cosh-based Soft (α={alpha1})', linewidth=2, alpha=0.8)
    plt.plot(m_minus_theta, cosh_soft2, 'g:', label=f'Cosh-based Soft (α={alpha2})', linewidth=2, alpha=0.8)
    
    plt.xlabel('Membrane Potential - Threshold (m - θ)')
    plt.ylabel('Spike Output')
    plt.title('Forward Pass: Hard vs Soft Approximations')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='right')
    
    # Add text box explaining the relationship
    textstr = 'Note: Cosh surrogate uses hard spikes (Heaviside)\nin forward pass, but soft gradients in backward pass'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid Tkinter issues
    print(f"Surrogate comparison saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fast-sigmoid vs cosh surrogate gradients.")
    parser.add_argument("--save-path", default="results/plots/cosh_surrogate_comparison.png")
    parser.add_argument("--alpha1", type=float, default=10.0)
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--alpha2", type=float, default=5.0)
    parser.add_argument("--beta2", type=float, default=1.0)
    args = parser.parse_args()

    plot_surrogate_comparison(
        save_path=args.save_path,
        alpha1=args.alpha1,
        beta1=args.beta1,
        alpha2=args.alpha2,
        beta2=args.beta2,
    )
