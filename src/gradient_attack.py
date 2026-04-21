import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from copy import deepcopy

class AdversarialMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128, 64, 32)):
        super(AdversarialMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(h))
            # layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def p_rule_penalty(logits, sensitive_attr):
    """
    Approximates Demographic Parity penalty.
    Minimizes squared difference between mean prediction for A=1 and A=0.
    """
    probs = torch.softmax(logits, dim=1)[:, 1]
    mask_1 = (sensitive_attr == 1)
    mask_0 = (sensitive_attr == 0)
    
    if mask_1.sum() > 0 and mask_0.sum() > 0:
        mean_1 = probs[mask_1].mean()
        mean_0 = probs[mask_0].mean()
        return (mean_1 - mean_0) ** 2
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

def shap_penalty(model, x, target_indices):
    """
    Gradient-ratio penalty targeting SHAP.
    Maximizes gradient magnitude w.r.t targeted sensitive features.
    """
    if not target_indices:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
        
    x.requires_grad_(True)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[:, 1]
    
    grads = torch.autograd.grad(outputs=probs, inputs=x, 
                                grad_outputs=torch.ones_like(probs), 
                                create_graph=True, retain_graph=True)[0]
    
    target_mask = torch.zeros(x.shape[1], dtype=torch.bool)
    target_mask[target_indices] = True
    
    g_target = grads[:, target_mask].abs().mean()
    g_other = grads[:, ~target_mask].abs().mean()
    
    # We want to minimize (g_other - g_target) to effectively inflate target feature grad
    return g_other - g_target

def lime_penalty(model, x, target_indices):
    """
    Directional contrast penalty for LIME.
    Maximizes output shift when target features are perturbed.
    """
    if not target_indices:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
        
    x_pert = x.clone().detach()
    # Add random noise/flip to mimick finite bin-jump mechanism of LIME
    for idx in target_indices:
        x_pert[:, idx] = x_pert[:, idx] + torch.randn_like(x_pert[:, idx]) * 1.5
        
    logits_orig = model(x)
    probs_orig = torch.softmax(logits_orig, dim=1)[:, 1]    
    logits_pert = model(x_pert)
    probs_pert = torch.softmax(logits_pert, dim=1)[:, 1]
    
    mse_diff = torch.nn.functional.mse_loss(probs_orig, probs_pert)
    # We want to maximize this difference
    return -mse_diff

def train_adversarial_model(
    X_train, y_train, A_train, 
    X_val, y_val, 
    lambda_fair=1.0, 
    lambda_shap=0.0, 
    lambda_lime=0.0, 
    target_indices=None,
    max_epochs=1000, 
    patience=50, 
    lr=0.001,
    batch_size=256,
    verbose=True
):
    """
    Trains a PyTorch MLP using early stopping and dual-penalties + fairness constraint.
    """
    if target_indices is None:
        target_indices = []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    A_t = torch.tensor(A_train, dtype=torch.float32).to(device)
    
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_t, y_t, A_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AdversarialMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y, batch_A in dataloader:
            optimizer.zero_grad()
            
            # Predict
            logits = model(batch_X)
            
            # Classification Loss
            loss_ce = criterion(logits, batch_y)
            
            # Penalties
            loss_fair = p_rule_penalty(logits, batch_A) if lambda_fair > 0 else torch.tensor(0.0)
            
            if lambda_shap > 0:
                loss_shap = shap_penalty(model, batch_X.clone(), target_indices)
            else:
                loss_shap = torch.tensor(0.0)
                
            if lambda_lime > 0:
                loss_lime = lime_penalty(model, batch_X.clone(), target_indices)
            else:
                loss_lime = torch.tensor(0.0)
                
            # Total Loss
            total_loss = loss_ce + lambda_fair * loss_fair + lambda_shap * loss_shap + lambda_lime * loss_lime
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        # Validation Step
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Create scikit-learn wrapper for easy integration with SHAP/LIME
    class PyTorchWrapper:
        def __init__(self, pytorch_model):
            self.model = pytorch_model
            self.model.eval()
            self.classes_ = np.array([0, 1])
            
        def predict_proba(self, X):
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            else:
                X_tensor = X.to(device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
            
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
            
    return PyTorchWrapper(model)
