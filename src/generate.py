import matplotlib.pyplot as plt
from src.utils import convert_text_to_feature
import torch
from src.model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 768
noise_dim = 100
embed_out_dim = 64
generator = Generator(channels=3, embed_dim=embed_dim, 
                          noise_dim=noise_dim, embed_out_dim=embed_out_dim).to(device)
generator.load_state_dict(torch.load('models/generator_bert.pth'))
generator.eval() 
sentence = 'colorful flower in a pool'
embeddings = convert_text_to_feature([str(sentence)])
noise = torch.randn(1, noise_dim, 1, 1, device=device)
pred = generator(noise, embeddings)
plt.imshow(pred[0].cpu().detach().permute(1, 2, 0))
plt.show()