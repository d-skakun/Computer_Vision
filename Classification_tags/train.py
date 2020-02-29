import os
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from datetime import datetime
from tensorboardX import SummaryWriter
from utils import get_lr
from download_images import PATH, TRAIN_PATH, VALID_PATH

BATCH_SIZE = 64
EPOCHS = 30
NUM_WORKERS = 8
IMG_SIZE = 224
RANDOM = 17
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

random_choice_transform = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3),
    transforms.ColorJitter(contrast=0.3),
    transforms.ColorJitter(saturation=0.3),
    transforms.RandomGrayscale(p=0.5),
])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        random_choice_transform,
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ]),
    'valid': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
}

data = {
    'train': datasets.ImageFolder(root=TRAIN_PATH, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(root=VALID_PATH, transform=data_transforms['valid'])
}

# Loader
train_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
valid_loader = DataLoader(data['valid'], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

# Variables
classes_name = data['train'].classes
classes_count = len(classes_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = datetime.now()
output_to_tensorboard = True
model_name = "resnet50"
best_acc = 0

# Model
model = models.resnet50(pretrained=True)

# Change the last layer
model.fc = nn.Linear(model.fc.in_features, classes_count)
model = model.to(device)

# Loss Function
loss_func = nn.CrossEntropyLoss()

# Optimizer
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=2, min_lr=1e-5)

# tensorboard
if output_to_tensorboard:
    log_dir = '../tensorboard/{}_{}'.format(model_name, now.strftime('%Y-%m-%d-%H-%M'))
    tensorboard_writer = SummaryWriter(log_dir)
else:
    tensorboard_writer = None

# Train and valid
for epoch in tqdm(range(1, EPOCHS+1), total=EPOCHS):
    train_loss, train_loss_avg, train_acc = 0, 0, 0
    correct, total, batch_idx = 0, 0, 0

    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        batch_idx += 1

        # Loss train
        train_loss += loss.item()
        train_loss_avg = train_loss/batch_idx

        # Accuracy train
        _, pred = torch.max(outputs.data, 1)
        correct += torch.sum(pred == labels.data)
        train_acc = float(100.0 * correct)/total

        # print('Train Loss {:.4f} | Acc={:.2f}% ({}/{})'.format(train_loss_avg, train_acc, correct, total))

    valid_loss, valid_loss_avg, valid_acc = 0, 0, 0
    correct, total, batch_idx = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for j, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            total += labels.size(0)
            batch_idx += 1

            # Loss valid
            valid_loss += loss.item()
            valid_loss_avg = valid_loss/batch_idx

            # Accuracy valid
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels.data)
            valid_acc = float(100.0 * correct)/total

        print('Epoch: {}/{}. Loss {:.4f} / {:.4f} | Acc {:.2f} / {:.2f} (train / valid)'
                            .format(epoch, EPOCHS, train_loss_avg, valid_loss_avg, train_acc, valid_acc))

    scheduler.step(valid_loss_avg)

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), os.path.join('./models', 'model_{}_{}.pt'.format(model_name, now.strftime("%d-%m-%Y"))))

    if output_to_tensorboard:
        tensorboard_writer.add_scalar('Learning rate', get_lr(optimizer), epoch)
        tensorboard_writer.add_scalar('Train/Loss', train_loss_avg, epoch)
        tensorboard_writer.add_scalar('Train/Accuracy', train_acc, epoch)
        tensorboard_writer.add_scalar('Validation/Val_Loss', valid_loss_avg, epoch)
        tensorboard_writer.add_scalar('Validation/Accuracy', valid_acc, epoch)


