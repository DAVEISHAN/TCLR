import torch
import random
import torch.nn as nn
import numpy as np

def global_local_temporal_contrastive(lsr,gdr, temperature):
    #lsr denotes local sparse-clip representation= representation of temporal slice of global clip
    #gdr denotes global dense-clip representation= representation of global(pooled) feature of local clip

    #lsr,gdr shape should be  [BS,4,128]
    similarity_matrix = torch.bmm(lsr, gdr.permute(0,2,1)) # [BS, 4, 4]
    # print(similarity_matrix)
    similarity_matrix = torch.cat((similarity_matrix, similarity_matrix.permute(0,2,1)),dim=0) # [BS*2, 4, 4]
    # print()
    # print(similarity_matrix)
    similarity_matrix = similarity_matrix.view(-1,4) # [BS*8, 4]
    # print()
    # print(similarity_matrix)
    # print()
    sample_lab = [0,1,2,3] 
    label = []
    for i in range(lsr.shape[0]*2):
        label.extend(sample_lab)
    label = torch.from_numpy(np.asarray(label)).long().cuda()

    similarity_matrix /= temperature

    loss = nn.functional.cross_entropy(similarity_matrix, label, reduction='sum')
    return loss/ (2*lsr.shape[0])

if __name__ == '__main__':
    BS = 40
    emb_size = 128
    lsr = nn.functional.normalize(torch.rand(BS,4, emb_size),dim=2).cuda()
    gdr = nn.functional.normalize(torch.rand(BS,4, emb_size),dim=2).cuda()
    loss = global_local_temporal_contrastive(lsr, gdr, 0.1)
    print(f'Loss is {loss}')


