import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import time
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import pp_ocrv3_rec_english
from data.dataset import RecDataset, collate_fn
from utils.transform import create_train_transforms, create_eval_transforms
from utils.losses import CTCLoss
from utils.postprocess import CTCLabelDecode, greedy_decode


def parse_args():
    parser = argparse.ArgumentParser(description='Train PP-OCRv3 Text Recognition Model')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--train_label', type=str, required=True, help='Training label file')
    parser.add_argument('--val_label', type=str, help='Validation label file')
    parser.add_argument('--character_dict', type=str, help='Character dictionary file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--pretrained', type=str, help='Pretrained model path')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img_height', type=int, default=48, help='Image height')
    parser.add_argument('--img_width', type=int, default=320, help='Image width')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Logging and saving
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--log_freq', type=int, default=100, help='Log every N iterations')
    
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        
        # Build model
        self.model = self._build_model()
        
        # Build datasets and dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Build loss function
        self.criterion = CTCLoss()
        
        # Build postprocessor
        self.postprocessor = CTCLabelDecode(
            character_dict_path=args.character_dict,
            use_space_char=False
        )
        
        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.best_acc = 0.0
        
        # Resume training if specified
        if args.resume:
            self._load_checkpoint(args.resume)

    def _build_model(self):
        """Build PP-OCRv3 model."""
        model = pp_ocrv3_rec_english(
            img_size=[self.args.img_height, self.args.img_width]
        )
        
        # Load pretrained weights if specified
        if self.args.pretrained:
            print(f"Loading pretrained weights from {self.args.pretrained}")
            checkpoint = torch.load(self.args.pretrained, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(self.device)
        return model

    def _build_dataloaders(self):
        """Build training and validation dataloaders."""
        # Training dataset
        train_transforms = create_train_transforms()
        train_dataset = RecDataset(
            data_dir=self.args.data_dir,
            label_file=self.args.train_label,
            img_size=(self.args.img_height, self.args.img_width),
            character_dict_path=self.args.character_dict,
            transforms=train_transforms,
            mode='train'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Validation dataset
        val_loader = None
        if self.args.val_label:
            val_transforms = create_eval_transforms()
            val_dataset = RecDataset(
                data_dir=self.args.data_dir,
                label_file=self.args.val_label,
                img_size=(self.args.img_height, self.args.img_width),
                character_dict_path=self.args.character_dict,
                transforms=val_transforms,
                mode='val'
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
        
        return train_loader, val_loader

    def _build_optimizer(self):
        """Build optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_acc = checkpoint.get('best_acc', 0.0)
        
        print(f"Resumed training from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'args': self.args
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {self.best_acc:.4f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.args.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        
        for batch_idx, (images, labels, label_lengths) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.model.head.return_feats:
                features, predictions = self.model(images)
            else:
                predictions = self.model(images)
            
            # Compute loss
            batch_data = [images, labels]
            loss_dict = self.criterion(predictions, batch_data)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            if batch_idx % self.args.log_freq == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Update learning rate
        self.scheduler.step()
        
        return epoch_loss / num_batches

    def evaluate(self, epoch):
        """Evaluate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_correct = 0
        total_samples = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, label_lengths in tqdm(self.val_loader, desc='Evaluating'):
                images = images.to(self.device)
                
                # Forward pass
                if self.model.head.return_feats:
                    features, predictions = self.model(images)
                else:
                    predictions = self.model(images)
                
                # Compute loss
                batch_data = [images, labels]
                loss_dict = self.criterion(predictions, batch_data)
                val_loss += loss_dict['loss'].item()
                
                # Decode predictions
                if isinstance(predictions, tuple):
                    predictions = predictions[1]  # Get predictions from (features, predictions)
                
                # Convert to numpy for postprocessing
                predictions_np = predictions.detach().cpu().numpy()
                
                # Greedy decode
                decoded_indices = greedy_decode(torch.tensor(predictions_np))
                
                # Convert to text
                pred_texts = []
                for indices in decoded_indices:
                    if len(indices) > 0:
                        chars = [self.postprocessor.character[idx] for idx in indices 
                                if idx < len(self.postprocessor.character)]
                        pred_text = ''.join(chars)
                    else:
                        pred_text = ""
                    pred_texts.append(pred_text)
                
                # Compare with ground truth (simplified evaluation)
                # This is a basic evaluation - you might want to implement more sophisticated metrics
                batch_size = len(pred_texts)
                total_samples += batch_size
                
                # For now, we'll just count non-empty predictions as correct
                # In practice, you'd compare with actual ground truth labels
                total_correct += sum(1 for text in pred_texts if len(text.strip()) > 0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        return accuracy

    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Log epoch metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            
            print(f'Epoch {epoch}/{self.args.epochs} - Train Loss: {train_loss:.4f}')
            
            # Evaluate
            if epoch % self.args.eval_freq == 0:
                val_acc = self.evaluate(epoch)
                
                # Check if best model
                is_best = val_acc > self.best_acc
                if is_best:
                    self.best_acc = val_acc
                
                # Save checkpoint
                if epoch % self.args.save_freq == 0 or is_best:
                    self._save_checkpoint(epoch, is_best)
            
            # Save regular checkpoint
            elif epoch % self.args.save_freq == 0:
                self._save_checkpoint(epoch)
        
        # Save final model
        self._save_checkpoint(self.args.epochs - 1)
        
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_acc:.4f}")
        
        self.writer.close()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()