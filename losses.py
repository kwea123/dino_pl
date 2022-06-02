import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import reduce


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, final_teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, final_teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * final_teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_output: (B*ncrops, out_dim)
        teacher_output: (B*2, out_dim)
        """
        student_out = student_output/self.student_temp
        student_out = student_out.chunk(self.ncrops) # global views + local views

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output-self.center)/temp, dim=-1)
        teacher_out = teacher_out.chunk(2) # global views

        total_loss = n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: # skip cases where student and teacher operate on the same view
                    continue
                loss = reduce(-q*F.log_softmax(student_out[v], dim=-1), 'b o -> b', 'sum')
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = reduce(teacher_output, 'b o -> 1 o', 'mean')

        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)