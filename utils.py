import torch 
import torch.nn as nn
import torch.nn.functional as F


def temperature_scaled_softmax(logits, temperature):
        return F.softmax(logits / temperature, dim=-1)

def get_pseudo_labels_from_logits(logits, P):
    batch_size = int(logits.shape[0]) // 10    
    similarity = temperature_scaled_softmax(logits, P['temp'])
    sim_global = similarity[0:batch_size] # score of global region
    s9 = torch.chunk(similarity[batch_size:], 9, dim=0) 
    sim_locals = torch.stack(s9, dim=0) # score of 9 local regions
    sim_locals = sim_locals.permute(1, 0, 2) # shape: (batch_size, 9, num_classes)
    eta = P['eta']
    alpha = torch.max(sim_locals, dim=1)[0]
    beta = torch.min(sim_locals, dim=1)[0]
    theta = torch.mean(sim_locals, dim=1)
    # gamma_i = 1 if alpha_i > eta else 0
    gamma = torch.where(alpha > eta, torch.tensor(1).to(P['device']), torch.tensor(0).to(P['device'])) 
    sim_ag = alpha * gamma + beta * (1 - gamma)
    sim_final = 1/ 2 * (sim_global + sim_ag)
    
    # define pseudo labels
    pseudo_labels = torch.zeros_like(sim_final)
    #positive pseudo label
    scalar_pos = torch.tensor(1, dtype=torch.float32).to(P['device'])
    pseudo_labels = torch.where(sim_final > P['threshold'], scalar_pos, pseudo_labels)

    return pseudo_labels
    