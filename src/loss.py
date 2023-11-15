import torch



class NTXentLoss(torch.nn.Module):
    def __init__(self, zeta=1e8):
        super().__init__()
        self.zeta = zeta

        self.criterion = torch.nn.CrossEntropyLoss()
        self.mask_buffer = {}
        self.label_buffer = {}

    def forward(self, z, temperature):
        """ z must have the shape of torch.cat([z_i, z_j], dim=0) """
        batch_size = z.shape[0]//2

        if batch_size not in self.mask_buffer.keys():
            self.mask_buffer[batch_size] = self.zeta*torch.eye(batch_size*2).to(z.device)
        m_sim = self.sim(z)/temperature - self.mask_buffer[batch_size].to(z.device)  # -zeta on diag -> 0 after exp in cross entropy
        
        if batch_size not in self.label_buffer.keys():
            self.label_buffer[batch_size] = torch.cat([torch.arange(batch_size, batch_size*2), torch.arange(0, batch_size)]).to(z.device)
        i_pos = self.label_buffer[batch_size]

        return self.criterion(m_sim, i_pos)
    
    def sim(self, z):
        z_norm = torch.nn.functional.normalize(z)
        m_sim = torch.mm(z_norm, z_norm.t())

        return m_sim
