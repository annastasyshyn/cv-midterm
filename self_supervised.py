
import torch
import torch.nn as nn
import albumentations as A



class SelfSupervisedInstanceEmbedder(nn.Module):
    """
    MASA adapter like instance embedder

    Define a list of strong augmentations of the same instance to drive
    the same object's appearance closer and yet far from the other objects
    """
    def __init__(self, backbone: nn.Module, transforms) -> None:
        
        super().__init__()

        self.backbone = backbone
        self.transforms = transforms
    
    def contrastive_data_split(self, x:torch.Tensor, y:torch.Tensor):
        """
        Create positive and negative data splits here
        
        X -- class instances rois
        y -- class instance ids

        """
        
        x = self.transforms(x)
        
        

    @staticmethod
    def constrastive_loss(positive_instances:torch.Tensor, negative_instances:torch.Tensor):
        ...
