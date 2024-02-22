import torch
import torch.nn as nn
from torchmetrics import Accuracy

import lightning as pl # Pytorch lightning is a wrapper for pytorch that makes it easier to train models
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

NUM_CLASSES = 6
def make_trainer(subdir, max_epochs, earlyStoppingPatience=10, refreshRate=1):
    # Initialize checkpoint callback to save the best model using validation loss
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"checkpoints_section/{subdir}/",
            save_top_k=1,        
            mode="min",
            every_n_epochs=1
        )

    # Create customized progress bar theme (Optional)
    progress_bar_task = RichProgressBar(refresh_rate=refreshRate, leave=False,
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82"
        )
    )

    # Call the Trainer and train the model
    early_stopping = EarlyStopping('val_loss', patience = earlyStoppingPatience, mode = 'min')
    trainer_task = pl.Trainer(
        accelerator="auto",
        devices=1, #if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=max_epochs,
        callbacks=[progress_bar_task, checkpoint_callback,early_stopping],
        logger=CSVLogger(save_dir=f"logs_task/{subdir}/"),
    )

    return trainer_task

class FeatureMLP(pl.LightningModule):
    def __init__(self, trainloader, valloader,testloader, num_classes=NUM_CLASSES, learning_rate=1.5e-3):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        
        # You will need to define your fully connected layer:
        self.fc1 = nn.Linear(20,64)
        self.fc2 = nn.Linear(64,48)
        self.fc3 = nn.Linear(48, num_classes)

        self.dropout = nn.Dropout(0.16)

        self.leaky_relu = nn.LeakyReLU()
        
        # Define your accuracies        
        self.train_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        
        
    def forward(self, x):

        x = torch.flatten(x,start_dim=1) #Ensure flattened data
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)
        acc = self.train_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        train_dict = {"train_loss": loss,
                    "train_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(train_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)
        acc = self.val_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        val_dict = {"val_loss": loss,
                    "val_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(val_dict, prog_bar=True, on_step=False, on_epoch=True)
    
        

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        
        self.test_accuracy.update(preds, y)
        acc = self.test_accuracy.compute()
        # Record accuracy and loss
        # Log anything you think necessary
        test_dict = {"test_loss": loss,
                    "test_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(test_dict, prog_bar=True, on_step=False, on_epoch=True)

        return
    
    def predict_step(self, batch, batch_idx):
        x, y, z= batch #z is for passthrough values to track samples for analysis
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)

        return preds, logits, x, y, z

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),self.learning_rate)
        return optimizer

    ##########
    # DATA RELATED HOOKS
    ##########

    def train_dataloader(self):
        return self.trainloader
    def val_dataloader(self):
        return self.valloader
    def test_dataloader(self):
        return self.testloader
    
class ConvBlockCNN(nn.Module):
    def __init__(self, channels_in, channels_out,stride=1):
        super(ConvBlockCNN, self).__init__()     # Call constructor

        middle_channel = (channels_in+channels_out)//2
        
        self.conv1 = nn.Conv2d(in_channels=channels_in,out_channels=middle_channel, kernel_size=3,stride=stride)
        self.conv2 = nn.Conv2d(in_channels=middle_channel,out_channels=channels_out, kernel_size=3,stride=stride)
        self.activ = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)

        x = self.pool(x)
        x = self.activ(x)

        x = self.conv2(x)
        x = self.activ(x)
        return x

class ImageCNN(pl.LightningModule):
    
    def __init__(self, trainloader, valloader, testloader, num_classes=NUM_CLASSES, learning_rate=1e-3):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # Initialise at least 2 convolutional blocks with
        self.conv_blk1 = ConvBlockCNN(3, 8,stride = 2)
        self.conv_blk2 = ConvBlockCNN(8,16,stride = 2)
        
        # Other Layers
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=20)
        self.dropout = nn.Dropout(0.1)
        
        # You will need to define your fully connected layer:
        self.fc1 = nn.Linear(16*20*20,100)
        self.fc2 = nn.Linear(100,64)
        self.fc3 = nn.Linear(64, num_classes)

        self.leaky_relu = nn.LeakyReLU()
        
        # Define your accuracies        
        self.train_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv_blk1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv_blk2(x)
        x = self.gap(x)

        x = torch.flatten(x,start_dim=1)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))

        return x
    
    def training_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)
        acc = self.train_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        train_dict = {"train_loss": loss,
                    "train_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(train_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)
        acc = self.val_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        val_dict = {"val_loss": loss,
                    "val_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(val_dict, prog_bar=True, on_step=False, on_epoch=True)
    
        

    def test_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)
        acc = self.test_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        test_dict = {"test_loss": loss,
                    "test_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(test_dict, prog_bar=True, on_step=False, on_epoch=True)

    
    def predict_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)

        return preds, logits, x, y

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),self.learning_rate)
        return optimizer

    ##########
    # DATA RELATED HOOKS
    ##########

    def train_dataloader(self):
        return self.trainloader
    def val_dataloader(self):
        return self.valloader
    def test_dataloader(self):
        return self.testloader


class ConvBlockECNN(nn.Module):

    def __init__(self, channels_in, channels_out,stride=1):
        super(ConvBlockECNN, self).__init__()     # Call constructor
        middle_channel = (channels_in+channels_out)//2
        print(middle_channel)
        self.conv1 = nn.Conv2d(in_channels=channels_in,out_channels=middle_channel, kernel_size=3,stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=middle_channel,out_channels=channels_out, kernel_size=3,stride=stride, padding=1)
        self.activ = nn.LeakyReLU()
        #self.batchnorm1 = nn.BatchNorm2d(middle_channel)
        #self.batchnorm2 = nn.BatchNorm2d(channels_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)

        x = self.conv2(x)
        x = self.activ(x)
        return x
    
class EncodedCNN(pl.LightningModule):
    
    def __init__(self, trainloader,valloader,testloader,num_classes=NUM_CLASSES, learning_rate=5e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        M = 32 #Number of channels between the two ConvBlocks
        self.batch_norm1 = nn.BatchNorm2d(19)
        self.batch_norm2 = nn.BatchNorm2d(M)

        
        # Initialise at least 2 convolutional blocks with
        self.conv_blk1 = ConvBlockECNN(19, M)
        self.conv_blk2 = ConvBlockECNN(M, 64)
        
        # You can use other layers too, feel free to define them here
        # self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=10)
        self.dropout = nn.Dropout(0.3)
        
        # You will need to define your fully connected layer:
        self.fc1 = nn.Linear(64*10*10,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, num_classes)

        self.leaky_relu = nn.LeakyReLU()
        
        # Define your accuracies        
        self.train_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass",num_classes=num_classes)
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.conv_blk1(x)
        x = self.batch_norm2(x)

        x = self.dropout(x)
        x = self.conv_blk2(x)
        x = self.gap(x)

        x = torch.flatten(x,start_dim=1)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))

        return x
    
    def training_step(self, batch, batch_idx):
        x, y, _= batch
        # print(x.shape,x)
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)
        acc = self.train_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        train_dict = {"train_loss": loss,
                    "train_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(train_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _= batch
        # print(x.shape,x)
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)
        acc = self.val_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        val_dict = {"val_loss": loss,
                    "val_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(val_dict, prog_bar=True, on_step=False, on_epoch=True)
    
        

    def test_step(self, batch, batch_idx):
        x, y, _= batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)
        acc = self.test_accuracy.compute()

        # Record accuracy and loss
        # Log anything you think necessary
        test_dict = {"test_loss": loss,
                    "test_acc": acc,
                    "epoch": self.current_epoch}
        
        self.log_dict(test_dict, prog_bar=True, on_step=False, on_epoch=True)

        return preds, logits, x, y
    
    def predict_step(self, batch, batch_idx):
        x, y, z = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)

        return preds, logits, x, y, z

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),self.learning_rate, weight_decay=1e-4)
        return optimizer

    ##########
    # DATA RELATED HOOKS
    ##########

    def train_dataloader(self):
        return self.trainloader
    def val_dataloader(self):
        return self.valloader
    def test_dataloader(self):
        return self.testloader



