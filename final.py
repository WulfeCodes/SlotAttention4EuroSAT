import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import itertools
import json
from datetime import datetime
import gc
import numpy as np
import copy  # <--- NEW: Essential for saving model weights

# Import from Scouter
from scouter.sloter.utils.slot_attention import SlotAttention
from scouter.sloter.utils.position_encode import build_position_encoding

# --- Custom Exception for Flow Control ---
class NaNError(Exception):
    pass

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ScouterSlotAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, slots_per_class=1, hidden_dim=128, 
                 slot_iters=3, dropout=0.1, lambda_value=0.1):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            features_only=True,
            drop_rate=dropout
        )
        
        feature_info = self.backbone.feature_info
        self.backbone_dim = feature_info[-1]['num_chs']
        
        self.conv1x1 = nn.Conv2d(self.backbone_dim, hidden_dim, kernel_size=1, stride=1)
        
        # GroupNorm to stabilize feature magnitude
        self.norm = nn.GroupNorm(8, hidden_dim)
        
        self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
        
        self.slot_attention = SlotAttention(
            num_classes=num_classes,
            slots_per_class=slots_per_class,
            dim=hidden_dim,
            iters=slot_iters,
            vis=False,
            loss_status=1,
            power=1
        )
        
        self.lambda_value = lambda_value
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        
    def forward(self, x, target=None, criterion=None):
        features = self.backbone(x)
        feature_map = features[-1]
        
        x = self.conv1x1(feature_map)
        x = torch.relu(x)
        x = self.norm(x)
        
        pe = self.position_emb(x)
        x_pe = x + pe
        
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1).permute(0, 2, 1)
        x_pe_flat = x_pe.reshape(b, c, -1).permute(0, 2, 1)
        
        logits, attn_loss = self.slot_attention(x_pe_flat, x_flat)
        
        # Aggregate slots if slots_per_class > 1
        if self.slots_per_class > 1:
             logits = logits.view(b, self.num_classes, self.slots_per_class).max(dim=2)[0]

        # Safety for NaN logits (though we want to catch this early)
        if torch.isnan(logits).any():
             pass 

        if target is not None:
            nll_loss = criterion(logits, target)
            total_loss = nll_loss + self.lambda_value * attn_loss
            return logits, total_loss, nll_loss, attn_loss
        
        return logits
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        slot_params = sum(p.numel() for p in self.slot_attention.parameters())
        other_params = total_params - backbone_params - slot_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'backbone': backbone_params,
            'backbone_trainable': backbone_trainable,
            'slot_attention': slot_params,
            'projection_layers': other_params
        }


def create_dataloaders(data_dir, batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = EuroSATDataset(data_dir, transform=None)
    total_size = len(full_dataset)
    
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    # Use generator for reproducibility
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = EuroSATDataset(data_dir, transform=transform_train)
    val_dataset = EuroSATDataset(data_dir, transform=transform_eval)
    test_dataset = EuroSATDataset(data_dir, transform=transform_eval)
    
    train_loader = DataLoader(Subset(train_dataset, train_indices), 
                             batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_indices), 
                           batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(Subset(test_dataset, test_indices), 
                            batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_dataset.classes


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output, loss, _, _ = model(images, labels, criterion)
            
            if torch.isnan(loss):
                continue

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    acc = 100. * correct / total if total > 0 else 0
    return avg_loss, acc


def two_stage_train(model, train_loader, val_loader, criterion, device, 
                    stage1_epochs=30, stage2_epochs=20, stage1_lr=4e-4, stage2_lr=1e-4,
                    verbose=True):
    """
    Two-stage training with BEST MODEL CHECKPOINTING.
    """
    
    # Initialize best weights tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_overall_acc = 0.0
    best_stage1_acc = 0.0

    # ==========================
    # STAGE 1: Frozen Backbone
    # ==========================
    if verbose:
        print("  [Stage 1] Training slot attention with frozen backbone...")
    
    model.freeze_backbone()
    
    optimizer_stage1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr, weight_decay=0.01
    )
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1, T_max=stage1_epochs
    )
    
    for epoch in range(stage1_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer_stage1.zero_grad()
            logits, loss, nll_loss, attn_loss = model(images, labels, criterion)
            
            if torch.isnan(loss):
                raise NaNError(f"NaN loss detected at Stage 1, Epoch {epoch+1}, Batch {batch_idx}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_stage1.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        scheduler_stage1.step()
        
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # --- SAVE BEST WEIGHTS (STAGE 1) ---
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_stage1_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{stage1_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (Best: {best_overall_acc:.2f}%)")
    
    if verbose:
        print(f"  [Stage 1] Best Val Acc: {best_stage1_acc:.2f}%")

    # ==========================
    # INTERLUDE: Load Best Weights
    # ==========================
    # Before unfreezing, load the best weights from Stage 1. 
    # This prevents starting Stage 2 from a potentially overfitted/degraded state.
    model.load_state_dict(best_model_wts)
    
    # ==========================
    # STAGE 2: Unfrozen Backbone
    # ==========================
    if verbose:
        print("  [Stage 2] Fine-tuning entire model with unfrozen backbone...")
    
    model.unfreeze_backbone()
    
    optimizer_stage2 = torch.optim.AdamW(
        model.parameters(),
        lr=stage2_lr, weight_decay=0.01
    )
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage2, T_max=stage2_epochs
    )
    
    for epoch in range(stage2_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer_stage2.zero_grad()
            logits, loss, nll_loss, attn_loss = model(images, labels, criterion)
            
            if torch.isnan(loss):
                raise NaNError(f"NaN loss detected at Stage 2, Epoch {epoch+1}, Batch {batch_idx}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_stage2.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        scheduler_stage2.step()
        
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # --- SAVE BEST WEIGHTS (STAGE 2) ---
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{stage2_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (Best: {best_overall_acc:.2f}%)")
    
    if verbose:
        print(f"  [Stage 2] Best Overall Val Acc: {best_overall_acc:.2f}%")
        print("  Restoring best model weights for final testing...")
    
    # --- FINAL RESTORE ---
    # Load the absolute best weights found in either Stage 1 or Stage 2
    model.load_state_dict(best_model_wts)
    
    return best_stage1_acc, best_overall_acc


def grid_search(data_dir, device='cuda', stage1_epochs=30, stage2_epochs=20, save_top_n=3, max_retries=3):
    """Run grid search with auto-retry on NaN"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f'checkpoints_{timestamp}')
    checkpoint_dir.mkdir(exist_ok=True)
    
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir, batch_size=32
    )
    num_classes = len(classes)
    print(f"Loaded dataset: {num_classes} classes, {len(train_loader.dataset)} train samples")
    print(f"Max Retries per experiment: {max_retries}\n")
    
    # Define hyperparameter grid
    param_grid = {
        'model_name': ['mobilevitv2_075.cvnets_in1k'],
        'hidden_dim': [256], 
        'slot_iters': [5,3], 
        'slots_per_class': [1], 
        'lambda_value': [0.1, 0.05], 
        'dropout': [0.1,0.2], 
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_experiments = len(combinations)
    print(f"Total experiments to run: {total_experiments}")
    print("="*80)
    
    results = []
    criterion = nn.CrossEntropyLoss()
    progressive_results_file = checkpoint_dir / 'progressive_results.json'
    
    for idx, params in enumerate(combinations, 1):
        exp_name = (f"{params['model_name'].split('.')[0]}_"
                    f"hd{params['hidden_dim']}_"
                    f"iter{params['slot_iters']}_"
                    f"spc{params['slots_per_class']}_"
                    f"lam{params['lambda_value']}_"
                    f"drop{params['dropout']}")
        
        print(f"\n[{idx}/{total_experiments}] {exp_name}")
        print("-" * 80)
        
        success = False
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  ⚠ Retry Attempt {attempt + 1}/{max_retries}...")
            
            try:
                # Create model
                model = ScouterSlotAttentionClassifier(
                    model_name=params['model_name'],
                    num_classes=num_classes,
                    slots_per_class=params['slots_per_class'],
                    hidden_dim=params['hidden_dim'],
                    slot_iters=params['slot_iters'],
                    dropout=params['dropout'],
                    lambda_value=params['lambda_value']
                )
                model = model.to(device)
                
                if attempt == 0:
                    param_counts = model.count_parameters()
                    print(f"  Model parameters: {param_counts['total']:,} total")
                
                # Train (model is updated in-place to the BEST weights)
                val_acc_stage1, val_acc_stage2 = two_stage_train(
                    model, train_loader, val_loader, criterion, device,
                    stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
                    stage1_lr=4e-4, stage2_lr=1e-4,
                    verbose=True
                )
                
                # Test - Now guaranteed to be using the BEST weights
                model.eval()
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                
                print(f"✓ Success! Test Acc: {test_acc:.2f}% (Evaluated on Best Model)")
                
                result = {
                    'experiment': exp_name,
                    'params': params,
                    'val_acc_stage1': val_acc_stage1,
                    'val_acc_stage2': val_acc_stage2,
                    'test_acc': test_acc,
                    'param_counts': param_counts,
                    'model_state': {
                        'state_dict': model.state_dict(),
                        'model_name': params['model_name']
                    }
                }
                results.append(result)
                success = True
                
                del model
                torch.cuda.empty_cache()
                gc.collect()
                break

            except NaNError as e:
                print(f"  ✗ Failed: {e}")
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ CRITICAL FAILURE: {e}")
                import traceback
                traceback.print_exc()
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                gc.collect()
        
        if not success:
            print(f"  ☠ Skipping configuration after {max_retries} failures.")
            results.append({
                'experiment': exp_name,
                'params': params,
                'test_acc': None,
                'error': "Max retries exceeded (NaN or other error)"
            })
        
        results_to_save = []
        for r in results:
            r_copy = r.copy()
            if 'model_state' in r_copy:
                del r_copy['model_state']
            results_to_save.append(r_copy)
        
        with open(progressive_results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

    # Save Final Checkpoints
    valid_results = [r for r in results if r.get('test_acc') is not None]
    valid_results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    print("\n" + "="*80)
    print(f"SAVING TOP {save_top_n} MODEL CHECKPOINTS")
    print("="*80)
    
    for i, result in enumerate(valid_results[:save_top_n], 1):
        if 'model_state' in result:
            checkpoint_path = checkpoint_dir / f'rank_{i}_{result["experiment"]}.pth'
            checkpoint = {
                'model_state_dict': result['model_state']['state_dict'],
                'experiment_name': result['experiment'],
                'params': result['params'],
                'test_acc': result['test_acc']
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved Rank {i} (Acc: {result['test_acc']:.2f}%)")
            del result['model_state'] 

    with open(checkpoint_dir / f'final_results_{timestamp}.json', 'w') as f:
        for r in results:
            if 'model_state' in r: del r['model_state']
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    results = grid_search(
        data_dir='./EuroSAT',
        device='cuda',
        stage1_epochs=55,
        stage2_epochs=45,
        save_top_n=3,
        max_retries=10 
    )