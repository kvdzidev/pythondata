# pythondata
zad dodatkowe lab10:
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

przeksztalc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dane_treningowe = datasets.MNIST(root='./dane', train=True, download=True, transform=przeksztalc)
ladowarka_danych = torch.utils.data.DataLoader(dane_treningowe, batch_size=64, shuffle=True)

class SiecAutoenkoder(nn.Module):
    def __init__(self):
        super(SiecAutoenkoder, self).__init__()
        self.koder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.dekoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, wejscie):
        zakodowane = self.koder(wejscie)
        odtworzone = self.dekoder(zakodowane)
        return odtworzone

model = SiecAutoenkoder()
funkcja_straty = nn.MSELoss()
optymalizator = optim.Adam(model.parameters(), lr=0.001)

epoki = 10
urzadzenie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(urzadzenie)

for epoka in range(epoki):
    model.train()
    suma_strat = 0
    for obrazy, _ in ladowarka_danych:
        obrazy = obrazy.to(urzadzenie)
        optymalizator.zero_grad()
        wyjscie = model(obrazy)
        strata = funkcja_straty(wyjscie, obrazy)
        strata.backward()
        optymalizator.step()
        suma_strat += strata.item()
    
    srednia_strata = suma_strat / len(ladowarka_danych)
    print(f'Epoka [{epoka+1}/{epoki}], Strata: {srednia_strata:.4f}')

model.eval()
iterator_danych = iter(ladowarka_danych)
oryginaly, _ = next(iterator_danych)
oryginaly = oryginaly[:8].to(urzadzenie)
rekonstrukcje = model(oryginaly).cpu().detach()

fig, osie = plt.subplots(2, 8, figsize=(12, 4))
for i in range(8):
    osie[0, i].imshow(oryginaly[i].cpu().squeeze(), cmap='gray')
    osie[0, i].axis('off')
    osie[1, i].imshow(rekonstrukcje[i].squeeze(), cmap='gray')
    osie[1, i].axis('off')
plt.show()
