import torch
from torch.distributions import multinomial
from d2l import torch as d2l
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1, fair_probs).sample())

# counts = multinomial.Multinomial(1, fair_probs).sample()
# counts /= 10
# print(counts)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0) # 沿着轴0求和
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die="+str(i+1)+")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.xlabel('Groups of experiments')
plt.ylabel('Estimated probability')
plt.legend()
plt.show()


