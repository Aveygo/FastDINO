import torch
import torch.nn.functional as F
from model import Whisper

class DINO1D(torch.nn.Module):
    def __init__(self, primary_arch: Whisper, device: torch.device):
        super(DINO1D, self).__init__()
        self.primary = primary_arch.to(device)
        self.distance = torch.nn.CosineSimilarity()

    def distillation_loss(self, x, tau_s, tau_t):
        # primary_output [b, c, width]
        
        with torch.no_grad():
            teacher = self.primary.encoder(x)
            teacher_features = teacher.mean(-1)   
            #print(teacher.shape, teacher.mean(-1).shape, teacher.mean(0).shape) 
            teacher_output = teacher_features - teacher_features.mean(0)
            teacher_output = teacher_output.unsqueeze(-1)
            #print(teacher_output.shape)
        
        teacher_output = teacher_output.detach()
        student_output = self.primary.encoder(x)

        student_probs = F.log_softmax(student_output / tau_s, dim=1)
        teacher_probs = F.softmax(teacher_output / tau_t, dim=1)
        loss = - (teacher_probs * student_probs).sum(dim=1).mean()
        return loss
    
if __name__ == "__main__":
    from tqdm import tqdm
    import os
    from dataset import MusicDataset
    from torch.utils.data import DataLoader
    
    
    data_loader = DataLoader(MusicDataset("/home/greg/Datasets/music/mel_features"), batch_size=64, shuffle=True)
    model = Whisper()
    dino = DINO1D(model, "cuda")
    opt = torch.optim.Adam(dino.parameters(), lr=1e-4)
    
    if os.path.exists("/home/greg/Models/fastdino.pth"):
        checkpoint = torch.load("/home/greg/Models/fastdino.pth")
        dino.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
    
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        pbar = tqdm(data_loader)
        for idx, x in enumerate(pbar):
            

            loss = dino.distillation_loss(x.cuda(), 0.9, 0.04)
            #torch.nn.utils.clip_grad_norm_(dino.parameters(), 1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            
            pbar.set_description(f"Loss: {loss.item():.6f}")
        
        torch.save({
            "model": dino.state_dict(),
            "opt": opt.state_dict()
        }, "/home/greg/Models/fastdino.pth")