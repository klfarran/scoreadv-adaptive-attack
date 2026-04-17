import argparse
import torch
import torchattacks
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# for arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--attack",
    type=str,
    choices=["pgd", "auto", "fgsm", "cw"],
    required=True,
    help="Attack type: pgd, auto, fgsm, or cw"
)
args = parser.parse_args()


# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# N is num samples we want total
N = 7_500

train_size = 6_250
val_size   = 625
test_size  = 625

# reproducibility
# currently uses same images accross all 4 attacks- could change that by 
# assigning a different seed per attack if needed 
g = torch.Generator()
g.manual_seed(42)

#all_indices = torch.randperm(len(full_dataset), generator=g)
#subset_indices = all_indices[:N]

#train_idx = subset_indices[:train_size]
#val_idx   = subset_indices[train_size:train_size + val_size]
#test_idx  = subset_indices[train_size + val_size:]

# indices for train set
train_indices = torch.randperm(len(train_dataset), generator=g)
train_idx = train_indices[:train_size]

# indices for the test set 
test_indices = torch.randperm(len(test_dataset), generator=g)
val_idx  = test_indices[:val_size]
test_idx = test_indices[val_size : val_size + test_size]

train_set = Subset(train_dataset, train_idx)
val_set = Subset(test_dataset, val_idx)
test_set = Subset(test_dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=128, shuffle=False) 
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False) 

# model- ResNet-32, pretrained on CIFAR-10 
# chose 32 as it is a fair middleground between the weakest and strongest ResNet models available
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar10_resnet32",
    pretrained=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# choose attack type 
if args.attack == "pgd":
    atk = torchattacks.PGD(
        model,
        eps=8/255,
        alpha=2/255,
        steps=10
    )
    atk.set_normalization_used(mean=mean, std=std)
    
elif args.attack == "auto":
    atk = torchattacks.AutoAttack(
        model,
        norm="Linf",
        eps=8/255,
        version="standard"
    )
    atk.set_normalization_used(mean=mean, std=std)
    
elif args.attack == "fgsm":
    atk = torchattacks.FGSM(
    model,
    eps=8/255
    )
    atk.set_normalization_used(mean=mean, std=std)
    
elif args.attack == "cw":
    atk = torchattacks.CW(
      model,
      c=1e-4,          # confidence / tradeoff parameter
      kappa=0,         # confidence margin (0 = standard)
      steps=1000,      # optimization steps
      lr=0.01          # learning rate
    )
    atk.set_normalization_used(mean=mean, std=std)
    
attack_name = args.attack 


# denormalization
def denormalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return x * std + mean
    

# make sure output is acceptable input for diffusion model
def generate_diffusion_tensor(loader):
    adv_imgs_out = []
    clean_imgs_out = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        adv_imgs = atk(images, labels)
        
        with torch.no_grad():
            # process adversarial
            adv_imgs = denormalize(adv_imgs, mean, std)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            adv_imgs = adv_imgs * 2.0 - 1.0
            
            # process clean 
            clean_imgs = denormalize(images, mean, std)
            clean_imgs = torch.clamp(clean_imgs, 0.0, 1.0)
            clean_imgs = clean_imgs * 2.0 - 1.0
        
        adv_imgs_out.append(adv_imgs.cpu())
        clean_imgs_out.append(clean_imgs.cpu())
        
    return torch.cat(adv_imgs_out, dim=0), torch.cat(clean_imgs_out, dim=0)
    
    
train_adv, train_clean = generate_diffusion_tensor(train_loader)
val_adv, val_clean = generate_diffusion_tensor(val_loader)
test_adv, test_clean  = generate_diffusion_tensor(test_loader)

torch.save(train_adv, f"./{attack_name}_train.pt")
torch.save(val_adv,   f"./{attack_name}_val.pt")
torch.save(test_adv,  f"./{attack_name}_test.pt")

torch.save(train_clean, f"./{attack_name}_clean_train.pt")
torch.save(val_clean,   f"./{attack_name}_clean_val.pt")
torch.save(test_clean,  f"./{attack_name}_clean_test.pt")