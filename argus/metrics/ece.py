import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(
    probs: torch.Tensor, 
    labels: torch.Tensor, 
    confidences = None,
    n_bins: int = 40, 
    save_path: str = "reliability_diagram.png",
    title: str = "Reliability Diagram"
):
    """
    Generates and saves a Reliability Diagram (Accuracy vs. Confidence).
    """
    # 1. Setup bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # 2. Get predictions and confidences
    # For binary: confidence is max(p, 1-p), accuracy is (pred == label)
    if confidences is None:
        confidences = torch.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).long()

    
    accuracies = predictions.eq(labels).float()
    
    bin_accs = []
    bin_confs = []
    bin_counts = []

    # 3. Calculate metrics per bin
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            bin_accs.append(accuracies[in_bin].mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
            bin_counts.append(prop_in_bin)
        else:
            bin_accs.append(0)
            bin_confs.append((lower + upper).item() / 2)
            bin_counts.append(0)

    # 4. Plotting
    plt.figure(figsize=(8, 8))
    
    # The "Ideal" line
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
    
    # The actual accuracy bars
    # Using 'edge' alignment to match the bin boundaries
    plt.bar(bin_lowers.numpy(), bin_accs, width=1/n_bins, alpha=0.8, 
            edgecolor="black", color="blue", align='edge', label="Accuracy")
    
    # Optional: Gap visualization (Uncertainty/Calibration Error)
    # This shows the 'gap' between the bar and the diagonal
    plt.bar(bin_lowers.numpy(), [abs(a - c) for a, c in zip(bin_accs, bin_confs)], 
            bottom=bin_accs, width=1/n_bins, alpha=0.3, 
            edgecolor="red", color="red", hatch='/', align='edge', label="Gap")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0.5, 1) # Since binary confidence is always >= 0.5
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_im_reliability_diagram(
    y_true,
    y_probs,
    n_bins: int = 40, 
    save_path: str = "reliability_diagram.png",
    title: str = "Reliability Diagram"
    ):
    """
    y_true: label reali (0 o 1)
    y_probs: probabilità della classe positiva (es. model.predict_proba(X)[:, 1])
    """
    # Calcola la curva di calibrazione
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    
    # Linea di calibrazione perfetta
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibrazione Perfetta')
    
    # Curva del modello
    plt.plot(prob_pred, prob_true, marker='s', label='Modello (Classe Positiva)')
    
    # Istogramma delle predizioni per vedere lo sbilanciamento (opzionale ma utile)
    #plt.hist(y_probs, range=(0, 1), bins=n_bins, density=True, stacked=True, 
    #         alpha=0.3, color='blue', label='Distribuzione Prob.')

    plt.xlabel('Probabilità predetta (Classe Positiva)')
    plt.ylabel('Frazione di positivi reali')
    plt.title('Reliability Diagram: Focus su Classe Positiva')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
    
def binary_ece_from_logits(
    logits: torch.Tensor,   # shape (N,)
    labels: torch.Tensor,   # shape (N,)
    n_bins: int = 15,
):
    probs = torch.sigmoid(logits)
    confidences = torch.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).long()
    accuracies = predictions.eq(labels).float()

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            ece += prop * torch.abs(
                accuracies[in_bin].mean() - confidences[in_bin].mean()
            )

    return ece.item()

def binary_ece_from_probs(
    probs: torch.Tensor,   # shape (N,)
    labels: torch.Tensor,   # shape (N,)
    n_bins: int = 15,
):
    confidences = torch.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).long()
    accuracies = predictions.eq(labels).float()

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            ece += prop * torch.abs(
                accuracies[in_bin].mean() - confidences[in_bin].mean()
            )

    return ece.item()

def cbm_concept_ece(
    concept_logits: torch.Tensor,   # (N, C)
    concept_labels: torch.Tensor,   # (N, C)
    n_bins: int = 15,
):
    C = concept_logits.shape[1]
    eces = []

    for c in range(C):
        ece_c = binary_ece_from_logits(
            logits=concept_logits[:, c],
            labels=concept_labels[:, c],
            n_bins=n_bins
        )
        eces.append(ece_c)

    
    return {
        'eces': eces,
        'ece': sum(eces) / C
    }

def ece_from_probs(
    concept_probs: torch.Tensor,   # (N, C)
    concept_labels: torch.Tensor,   # (N, C)
    n_bins: int = 15,
):
    C = concept_probs.shape[1]
    eces = []

    for c in range(C):
        ece_c = binary_ece_from_probs(
            probs=concept_probs[:, c],
            labels=concept_labels[:, c],
            n_bins=n_bins
        )
        eces.append(ece_c)

    
    return {
        'eces': eces,
        'ece': sum(eces) / C
    }
