import torch
from lightly.loss.memory_bank import MemoryBankModule


class NNmemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16):
        super(NNmemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False):

        output, bank = super(NNmemoryBankModule, self).forward(output, labels, update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed)

        #nn
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)
        return nearest_neighbours

        #KNN (2)
        # index_nearest_neighbours = torch.topk(similarity_matrix, k=2, dim=1)[1]
        # nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.flatten())
        # nearest_neighbours = nearest_neighbours.view(output.size(0),2,-1)
        # k_1 = nearest_neighbours[:, 0, :]
        # k_2 = nearest_neighbours[:, 1, :]
        # return k_1, k_2

        #KNN(3)
        # index_nearest_neighbours = torch.topk(similarity_matrix, k=3, dim=1)[1]
        # nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.flatten())
        # nearest_neighbours = nearest_neighbours.view(output.size(0),3,-1)
        # k_1 = nearest_neighbours[:, 0, :]
        # k_2 = nearest_neighbours[:, 1, :]
        # k_3 = nearest_neighbours[:, 2, :]
        # return k_1, k_2, k_3

        # KNN(4)
        #index_nearest_neighbours = torch.topk(similarity_matrix, k=4, dim=1)[1]
        #nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.flatten())
        #nearest_neighbours = nearest_neighbours.view(output.size(0),4,-1)
        #k_1 = nearest_neighbours[:, 0, :]
        # k_2 = nearest_neighbours[:, 1, :]
        #k_3 = nearest_neighbours[:, 2, :]
        # k_4 = nearest_neighbours[:, 3, :]
        # return k_1, k_2, k_3, k_4

        #KNN(5)
        # index_nearest_neighbours = torch.topk(similarity_matrix, k=5, dim=1)[1]
        # nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.flatten())
        # nearest_neighbours = nearest_neighbours.view(output.size(0),5,-1)
        # k_1 = nearest_neighbours[:, 0, :]
        # k_2 = nearest_neighbours[:, 1, :]
        # k_3 = nearest_neighbours[:, 2, :]
        # k_4 = nearest_neighbours[:, 3, :]
        # k_5 = nearest_neighbours[:, 4, :]
        # return k_1, k_2, k_3,k_4, k_5


