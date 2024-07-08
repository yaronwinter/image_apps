import argparse
import train_cnn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EVAL_SET_FRACTION = 0.1

TRAIN_CNN = "train_cnn"

if __name__ == "__main__":
    print("Start Run")
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--prog_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--models_path", type=str, required=True)
    args = parser.parse_args()

    print(f"prog_name: {args.prog_name}")
    print(f"batch_size: {args.batch_size}")
    print(f"data_path: {args.data_path}")
    print(f"models_path: {args.models_path}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=transform)

    rands = [np.random.random() for i in range(len(trainset))]
    evalset = [trainset[i] for i in range(len(rands)) if rands[i] < EVAL_SET_FRACTION]
    trainset = [trainset[i] for i in range(len(rands)) if rands[i] >= EVAL_SET_FRACTION]

    print(f"train set: {len(trainset)}")
    print(f"test set: {len(testset)}")
    print(f"eval set: {len(evalset)}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.prog_name == TRAIN_CNN:
        train_cnn.train(trainloader, evalloader, testloader, args.models_path, "train_log.txt")
    else:
        raise Exception("Unknown program: " + args.prog_name)
