import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from .syn_vimeo_dataset import Syn_Vimeo_Dataset
from .syn_sintel_dataset import Syn_Sintel_Dataset

def fetch_dataloader_2E(args):
    train_vimeo = Syn_Vimeo_Dataset(root_dir=args.dataset_vimeo_dir, nframes=3, nexps=2, is_training=True)
    train_sintel_clean = Syn_Sintel_Dataset(root_dir=args.dataset_sintel_dir, dtype='clean', nframes=3, nexps=2, is_training=True)
    train_sintel_final = Syn_Sintel_Dataset(root_dir=args.dataset_sintel_dir, dtype='final', nframes=3, nexps=2, is_training=True)
    train_dataset = ConcatDataset([train_vimeo, train_sintel_clean, train_sintel_final])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_vimeo = Syn_Vimeo_Dataset(root_dir=args.dataset_vimeo_dir, nframes=3, nexps=2, is_training=False)
    val_loader = DataLoader(val_vimeo, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)    
    return train_loader, val_loader

def fetch_dataloader_3E(args):
    train_vimeo = Syn_Vimeo_Dataset(root_dir=args.dataset_vimeo_dir, nframes=5, nexps=3, is_training=True)
    train_sintel_clean = Syn_Sintel_Dataset(root_dir=args.dataset_sintel_dir, dtype='clean', nframes=5, nexps=3, is_training=True)
    train_sintel_final = Syn_Sintel_Dataset(root_dir=args.dataset_sintel_dir, dtype='final', nframes=5, nexps=3, is_training=True)
    train_dataset = ConcatDataset([train_vimeo, train_sintel_clean, train_sintel_final])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_vimeo = Syn_Vimeo_Dataset(root_dir=args.dataset_vimeo_dir, nframes=5, nexps=3, is_training=False)
    val_loader = DataLoader(val_vimeo, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)   
    return train_loader, val_loader


