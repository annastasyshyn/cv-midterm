
import torch
import torch.nn as nn
import albumentations as A



class SelfSupervisedInstanceEmbedder(nn.Module):
    """
    MASA adapter like instance embedder

    Define a list of strong augmentations of the same instance to drive
    the same object's appearance closer and yet far from the other objects
    """
    def __init__(self, backbone: nn.Module, transforms1, transforms2, temperature = 0.67) -> None:
        
        super().__init__()

        self.backbone = backbone
        self.transforms1 = transforms1
        self.transforms2 = transforms2

        self.temperature = temperature
    
    def contrastive_data_split(self, x:torch.Tensor):
        """
        Create positive and negative data splits here
        
        X -- class instances rois embeddings

        """
        
        transformed1 = self.transforms1(x)
        transformed2 = self.transforms2(x)

        ## fix the betching problem. might be moved foreward (and will be)
        transformed1 = torch.flatten(transformed1, start_dim=0, end_dim=0)
        transformed2 = torch.flatten(transformed2, start_dim=0, end_dim=0)

        transformed1 = nn.functional.normalize(transformed1, dim=1)
        transformed2 = nn.functional.normalize(transformed2, dim=1)


        similarities = (transformed1 @ transformed2) / self.temperature

        targets = torch.eye(similarities.shape[1])

        loss = nn.functional.cross_entropy(similarities, targets)

        return loss

        # for index, tr1 in enumerate(transformed1): ## for batch data

        #     SelfSupervisedInstanceEmbedder.constrastive_loss(positive_instances=torch.tensor(tr1, transformed2), 
        #                                                      negative_instances=transformed2[index:index])
        
        

    # @staticmethod
    # def constrastive_loss(positive_instances:torch.Tensor, negative_instances:torch.Tensor):
    #     """
    #     From the MASA paper contrantive loss: 
    #     -- pull the same object embeddings closer (positive pairs)
    #     -- push the different object embeddings from eac other (negative pairs)

    #     """

if __name__ == "__main__":
    B, H, W = 10, 50, 50
    input_batch = torch.randint(low=0, high=255, size=(B, H, W))

    
