import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_cuda_available():
    print(f"cuda available is: {torch.cuda.is_available()}")
    return torch.cuda.is_available()


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class NormalDistribution:
    def __init__(self, dict, device=None):
        self.sampler = torch.distributions.normal.Normal(dict["mu"], dict["std"])

    def sample(self, num_samples=1):
        return self.sampler.sample(sample_shape=[num_samples]).squeeze()


class NDParameterEstimator(nn.Module):
    def __init__(self):
        super(NDParameterEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, points: torch.Tensor):
        points, _ = torch.sort(points)
        pred = self.model(points)
        return pred


def main():
    check_cuda_available()

    # init model
    device = get_device()
    estimator = NDParameterEstimator().to(device)
    print(estimator)

    # optimization option
    learning_rate = 1e-5
    optimizer = torch.optim.SGD(estimator.parameters(), lr=learning_rate)

    # iterations (epochs)
    for ii in range(10000):

        # generate label (y)
        ii_rem = ii % 5
        print(f"data class is {ii_rem}")
        if ii_rem == 0:
            mu, std = torch.tensor([5.0]), torch.tensor([7.0])
        elif ii_rem == 1:
            mu, std = torch.tensor([-5.0]), torch.tensor([15.0])
        elif ii_rem == 2:
            mu, std = torch.tensor([2.0]), torch.tensor([3.3])
        elif ii_rem == 3:
            mu, std = torch.tensor([0.0]), torch.tensor([0.5])
        else:
            mu, std = torch.tensor([10.0]), torch.tensor([5.0])
        print(f" gt     : mu {mu.item():.2f}, std {std.item():.2f}")

        # generate data (x)
        ndg = NormalDistribution({"mu": mu, "std": std})
        sampled_val = ndg.sample(1000).to(device)
        sampled_mu, sampled_std = torch.mean(sampled_val), torch.std(sampled_val)
        print(f" sampled: mu {sampled_mu.item():.2f}, std {sampled_std.item():.2f}")

        # predict
        pred = estimator(sampled_val)
        print(f" pred   : mu {pred[0].item():.2f}, std {pred[1].item():.2f}")

        def L1_loss(pred):
            lmu = torch.abs(mu.to(device) - pred[0])
            lstd = torch.abs(std.to(device) - pred[1])
            diff = lmu + lstd
            return diff

        # learn parameters once
        loss = L1_loss(pred).to(device)
        print(f" loss: {loss.item():.3f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("")


if __name__ == "__main__":
    print(f"tuto1 started")
    main()
