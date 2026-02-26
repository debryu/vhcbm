from argus.utils.compute_ARGO_activations import get_argo_act
import torch
from torch.utils.data import DataLoader, TensorDataset
from CQA.models.glm_saga.elasticnet import IndexedTensorDataset, IndexedDataset, glm_saga
from loguru import logger
import psutil
import torch, os, copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def log_memory():
    # System RAM
    process = psutil.Process()
    print(f"RAM Usage: {process.memory_info().rss / 1024**2:.2f} MB")
    # GPU VRAM
    if torch.cuda.is_available():
        print(f"GPU Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Put this inside your training loop
log_memory()

class ArgoBackbone():
    def __init__(self, model_path):
        data_train, data_val, data_test = get_argo_act(model_path, model_path.split("-")[0])
        
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        
    def get_concept_preds(self, split):
        logger.warning(f"Getting {split} concepts variables")
        dataset = {'test': self.data_test, 'train': self.data_train, 'val': self.data_val}[split]
        n_samples = len(dataset)
        first_prob, first_pred, _, first_mean, first_std, first_gt = dataset[0]
        
        # 3. Pre-allocate tensors (much more memory efficient)
        all_preds = torch.zeros((n_samples, *first_pred.shape), dtype=first_pred.dtype)
        all_means = torch.zeros((n_samples, *first_mean.shape), dtype=first_mean.dtype)
        all_stds = torch.zeros((n_samples, *first_std.shape), dtype=first_std.dtype)
        all_gts = torch.zeros((n_samples, *first_gt.shape), dtype=first_gt.dtype)
        all_probs = torch.zeros((n_samples, *first_prob.shape), dtype=first_pred.dtype)
        
        # 4. Fill them in a loop
        for i in range(n_samples):
            prob, pred, unc, mean, std, gt = dataset[i]
            print(prob.shape)
            all_probs[i] = prob.detach().cpu()
            all_preds[i] = pred.detach().cpu()
            all_means[i] = mean.detach().cpu()
            all_stds[i] = std.detach().cpu()
            all_gts[i] = gt.detach().cpu()
            
        return {
            'concept_probs': all_probs,
            'concept_preds': all_preds,
            'concept_means': all_means,
            'concept_stds': all_stds,
            'concept_gts': all_gts
        }
            
class ArgoCBM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = ArgoBackbone(self.args.model_path)
        
    def forward(self, probs):
        logits = torch.logit(probs, eps=1e-8)
        preds = self.final_layer(logits)
        out_dict = {'concept_logits':logits, 'preds':preds, 'concept_probs':probs}
        return out_dict
    
    def load(self):
        # Load the final layer
        W_g = torch.load(os.path.join(self.args.load_dir, "W_g.pt"), map_location=self.args.device, weights_only=True)
        b_g = torch.load(os.path.join(self.args.load_dir, "b_g.pt"), map_location=self.args.device, weights_only=True)
        for i in range(W_g.shape[1]):
            print(W_g[1,i], i, i-42)
            
        #print(torch.sum(torch.abs(W_g[1,38:])), torch.sum(torch.abs(W_g[1,20:30])))
        self.final_layer = torch.nn.Linear(W_g.shape[1], W_g.shape[0]).to(self.args.device)
        self.final_layer.load_state_dict({"weight":W_g, "bias":b_g})
        
        row_to_plot = W_g[1, :].cpu().numpy()
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(row_to_plot)), row_to_plot, color='skyblue', edgecolor='navy')
        plt.title("Feature importance class 1")
        plt.xlabel("Input Feature Index")
        plt.ylabel("Weight Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.args.load_dir, "W_g.png"))
        return 
    
    def inference(self, x):
        device = self.args.device
        batch_size = getattr(self.args, "batch_size", 32)
        loader = DataLoader(x)
        logits = []
        classes = []
        for batch in tqdm(loader, desc="Eval"):
            batch = batch.to(self.args.device)
            preds = self.final_layer(batch)
            logits.append(preds)
            classes.append(torch.argmax(preds, dim=1))
        logits = torch.stack(logits, dim=0)
        classes = torch.stack(classes, dim=0)
        return {"logits": logits, "preds": classes}
    
    def train(self, train_x, val_x, test_x, train_y, val_y, test_y):
        # 1. Setup Hyperparameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = train_x.shape[1]
        #print(input_dim)
        num_classes = len(torch.unique(train_y))
        #print(num_classes)
        learning_rate = getattr(self.args, 'lr', 1e-3)
        epochs = getattr(self.args, 'epochs', 2000)
        patience = getattr(self.args, 'patience', 10)
        batch_size = getattr(self.args, 'batch_size', 1024)

        # 2. Prepare DataLoaders
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
        # 3. Initialize Model, Loss, and Optimizer
        model = torch.nn.Linear(input_dim, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Early Stopping Variables
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        # 4. Training Loop
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch_x, batch_y in tqdm(train_loader, desc="Training"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            logger.debug(f"Train loss {np.mean(train_losses)}")
            # 5. Validation Phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_x, v_y = v_x.to(device), v_y.to(device)
                    v_out = model(v_x)
                    val_loss += criterion(v_out, v_y).item()
            
            val_loss /= len(val_loader)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 6. Load best weights and extract W and b
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        W_g = model.weight.data.detach().cpu()
        b_g = model.bias.data.detach().cpu()

        # 7. Save results
        os.makedirs(self.args.save_dir, exist_ok=True)
        torch.save(W_g, os.path.join(self.args.save_dir, "W_g.pt"))
        torch.save(b_g, os.path.join(self.args.save_dir, "b_g.pt"))

        return self.args