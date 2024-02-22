import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from math import *
import torch
import torchvision
from torchvision import transforms
import os
import time

#GradCAM Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


NUM_CLASSES = 6
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

def norm_stats_2D(dataset):
    L=len(dataset)
    mean_total = np.zeros(3)
    std_total = np.zeros(3)
    for i in range(L):
        img = (dataset.__getitem__(i))[0].numpy()
        img = img.reshape(img.shape[0],img.shape[1]*img.shape[2])

        mean_total += np.mean(img,axis=1)
        std_total += np.std(img,axis=1)

    means = (mean_total/L).tolist()
    stds = (std_total/L).tolist()

    return means, stds    

def norm_stats(ds):
    ds = np.array(ds)
    means = (np.mean(ds,axis=0))
    stds = (np.std(ds,axis=0))
    print(means.shape,stds.shape)
    return means, stds

def plot_logs(metrics_dir, show=True, save=False, path=None):
    save_dir = path

    metrics_task = pd.read_csv(metrics_dir + "/metrics.csv")
    metrics_task.set_index("epoch", inplace=True)
    metrics_task = metrics_task.groupby(level=0).sum().drop("step", axis=1)

    # Plot using matplotlib
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.subplot(1,2,1)
    epochs = len(metrics_task)-1
    print(epochs)
    plt.plot(metrics_task["train_loss"][0:epochs-1])
    plt.plot(metrics_task["val_loss"][0:epochs-1])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train and Validation Loss")
    plt.legend(["Train", "Validation"]);

    plt.subplot(1,2,2)

    plt.plot(metrics_task["train_acc"][0:epochs-1])
    plt.plot(metrics_task["val_acc"][0:epochs-1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Train and Validation Accuracy")
    plt.legend(["Train", "Validation"]);

    if show:
        plt.show()
    if save:
        subj = metrics_dir.split('/')[4]
        print(subj)
        # exit()
        plt.savefig(f'{save_dir}/{subj}_{int(time.time())}.png')
        plt.clf()

def shapley(model, dataloader, feature_names, show, save=False, path="", samples_count=200):
    import shap
    save_dir = path
    time_dir = f'{save_dir}/shap_{int(time.time())}/'

    # Define function to wrap model to transform data to tensor
    f = lambda x: model(torch.from_numpy(x)).detach().numpy()

    # Convert my pandas dataframe to numpy
    data = next(iter(dataloader))[0].cpu()
    data = data.numpy()
    for _ in range(len(dataloader)-1):
        next_data = next(iter(dataloader))[0].cpu()
        data = np.vstack((data,next_data))
    
    print(np.shape(data))
    data = shap.sample(data,samples_count)
    # print(np.shape(data))
    # data = shap.kmeans(data,128)

    print(np.shape(data))

    explainer = shap.KernelExplainer(f, data)
    
    vals = explainer.shap_values(data)
    # vals = explainer(data)
    
    shap.summary_plot(vals, data, feature_names, show=show, max_display=40)
    if save:
            os.makedirs(time_dir)
            plt.savefig(f'{time_dir}/shap_summary.png')
    plt.clf()

    for i in range(NUM_CLASSES):
        shap.summary_plot(vals[i], data, feature_names, show=show, max_display=40, cmap=plt.get_cmap("tab20c"))
        # shap.plots.beeswarm(vals[i], max_display=40, show=False, color=plt.get_cmap("tab20c"))
        if show:
            plt.figure()
        if save:
            plt.savefig(f'{time_dir}/shap_class{i}.png')
        plt.clf()

def track_stats_2d(grad_max_sml, map_shape, map_layout, df_row):
    # map_shape = (len(map_layout),len(map_layout[0])) #map layout shape

    track_topk = torch.topk(grad_max_sml.reshape(-1),map_shape[0]*map_shape[1]) #Gets gradient values in order
    track_indices = track_topk.indices

    stat2 = []
    for max_idx in track_indices:
        max_pos = (max_idx//map_shape[1],max_idx%map_shape[1]) #Convert 1D index into 2D
        max_item = None
        try:
            max_item = map_layout[max_pos[0]][max_pos[1]] #Gets associated tile
        except IndexError:
            continue #Encoded CNN has padding and may not have associated map tile

        #Classfies Tile
        if max_item == " ": #If empty space or player
            found = False

            #Check if player
            p_idx= 1
            state = json.loads(df_row['state'])
            player_idx = df_row['player_idx']

            for player in [state['players'][player_idx], state['players'][(player_idx+1)%2]]:
                #Position
                player_pos = torch.tensor([player['position'][1],player['position'][0]]) #Corrected for rows then cols
                # print(player_pos,max_pos)
                if player_pos[0] == max_pos[0] and player_pos[1] == max_pos[1]:
                    found=True
                    break
                else:
                    p_idx+=1
            
            #Adds player or empty item
            if found == True:
                max_item = str(p_idx)
            else:
                max_item = "E"
        
        #Add any unique items
        if max_item not in stat2:
            stat2.append(max_item)
        
        #Stop when all(8) items are added
        if len(stat2)==8:
            break
    return stat2

def filenames(self, indices=[], basename=False):
        if indices: 
            # grab specific indices
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]
            
def saliency_ImageCNN(testset,test_dfs, model, subdir = "v2_2", index_list = None, plot=False, save_plot=False, track_stats=False):
    #Keep memory usage low by get each prediction individually and perform stats at runtime
    torchvision.datasets.ImageFolder.filenames = filenames   
    
    if index_list is None:
        index_list = range(len(testset)) #If none specified, iterate over entire dataset

    model.to(device) #ensures GPU/CPU consistentency
    square_stats = []

    for idx in index_list:
        # Retrieve the input image, true label
        item = testset.__getitem__(idx)
        image_original = item[0]
        label = item[1]
        
        #Obtain globally named index from filename
        global_index = testset.filenames([idx])[0].split("\\")[1].split(".")[0]
        # print(global_index)

        # Add a batch dimension to the input image (via .unsqueeze()) and set requires_grad to True for saliency analysis
        image = image_original.unsqueeze(0).clone().to(device)
        image.requires_grad = True

        # Compute the scores and gradients for the input image
        # To compute the scores, do a forward pass of the image and then take the argmax
        # Use this index to extract the score_max value from "scores"
        # Then perform a backward step so that it backpropagates the gradient
        model.eval()
        scores = model.forward(image)
        score_max_index = torch.argmax(scores)

        pred = score_max_index

        score_max = scores[0, score_max_index]
        score_max.backward() # Do the backward step here for the gradient calculation

        # Calculate the saliency map by finding the maximum absolute gradient values across channels
        # You can use .abs() and torch.max()
        grad = image.grad.to("cpu")
        grad_abs = torch.squeeze(grad.abs())
        grad_max = torch.max(grad_abs,dim=0).values

        #Get state from pandas files
        #Ensure corrected ids test set is used
        row = test_dfs.iloc[int(global_index)]
        map_layout = row['layout']
        map_layout = map_layout.replace("\'","\"").replace("1"," ").replace("2"," ")
        map_layout = json.loads(map_layout)

        map_shape = (len(map_layout),len(map_layout[0]))
        # print(map_shape)
        grad_abs_sml = transforms.functional.resize(grad_abs, map_shape, interpolation = torchvision.transforms.InterpolationMode.BICUBIC, antialias= True)
        grad_max_sml = torch.max(grad_abs_sml,dim=0).values


        #Unnormalise Image for display
        clean_img = torchvision.io.read_image(f"./data/imgs/test/{label}/{global_index}.png").permute(1,2,0)
        clean_img = np.array(clean_img)/255
        clean_img = np.clip(clean_img,0,1)

        #Downscaling and then upscaling gradient image for smoothened image
        scale_factor = 3
        grad_abs_med = transforms.functional.resize(grad_abs, (map_shape[0]*scale_factor, map_shape[1]*scale_factor), interpolation = torchvision.transforms.InterpolationMode.BICUBIC, antialias= True)
        grad_max_med = torch.max(grad_abs_med,dim=0).values
        up_down_grad_max = transforms.functional.resize(grad_max_med.unsqueeze(0), (grad_max.shape[0],grad_max.shape[1]), interpolation = torchvision.transforms.InterpolationMode.BICUBIC, antialias= True).squeeze().abs()
        up_down_max = torch.max(up_down_grad_max)
        up_down_grad_max = (up_down_grad_max/(up_down_max.item()))
        grad_max_image = show_cam_on_image(clean_img, up_down_grad_max, use_rgb=True)


        #For individual ranking stats
        if track_stats:
            stat2 = track_stats_2d(grad_max_sml, map_shape, map_layout, row)
            square_stats.append(stat2)
            

        # # Create a subplot to display the original image and saliency map side by side
        if plot or save_plot:
            # images = np.hstack((cam , cam_image))
            # grad_cam = Image.fromarray(images)
            actions_list = ["INT","Right","Left","Down","Up",]
            plt.close()
            plt.figure()

            # Create a subplot to display the original image and saliency map side by side
            plt.rcParams["figure.figsize"] = (10, 9)
            plt.subplot(2,2,1)
            plt.imshow(grad_max)
            plt.title(f"Saliency (Max)\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.axis('off')

            #Downsampled saliency
            plt.subplot(2,2,4)
            plt.imshow(grad_max_sml)
            plt.title(f"Downsized Saliency\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.axis('off')

            #Downsampled then upsampled overlay
            plt.subplot(2,2,3)
            plt.imshow(grad_max_image)
            plt.title(f"Smoothened Overlay\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.axis('off')

            #Original Image
            plt.subplot(2,2,2)
            plt.title(f"Original (Blue)\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.imshow((clean_img))
            plt.axis('off')

            plt.tight_layout()

            if plot:
                plt.show()
                print("Probabilities:",torch.nn.functional.softmax(scores))

            if save_plot:
                plt.savefig(fname=f"./results/{subdir}/{global_index}.png")


    return square_stats

def saliency_EncodedCNN(testset,test_dfs, model, subdir = "v2_2", index_list = None, plot=False, save_plot=False, track_stats=False):
    #Keep memory usage low by get each prediction individually and perform stats at runtime

    if index_list is None:
        index_list = range(len(testset)) #If none specified, iterate over entire dataset

    model.to(device) #ensures GPU/CPU consistentency
    square_stats = []

    for idx in index_list:
        # Retrieve the input image, true label
        item = testset.__getitem__(idx)
        image_original = item[0]
        label = item[1]
        global_index = item[2]
        # print(global_index)

        # Add a batch dimension to the input image (via .unsqueeze()) and set requires_grad to True for saliency analysis
        image = image_original.unsqueeze(0).clone().to(device)
        image.requires_grad = True

        # Compute the scores and gradients for the input image
        # To compute the scores, do a forward pass of the image and then take the argmax
        # Use this index to extract the score_max value from "scores"
        # Then perform a backward step so that it backpropagates the gradient
        model.eval()
        scores = model.forward(image)
        score_max_index = torch.argmax(scores)

        pred = score_max_index

        score_max = scores[0, score_max_index]
        score_max.backward() # Do the backward step here for the gradient calculation

        # Calculate the saliency map by finding the maximum absolute gradient values across channels
        # You can use .abs() and torch.max()
        grad = image.grad.to("cpu")
        grad_abs = torch.squeeze(grad.abs())
        grad_max = torch.max(grad_abs,dim=0).values

        #Get state from pandas files
        #Ensure corrected ids test set is used
        row = test_dfs.iloc[int(global_index)]
        map_layout = row['layout']
        map_layout = map_layout.replace("\'","\"").replace("1"," ").replace("2"," ")
        map_layout = json.loads(map_layout)

        map_shape = (len(map_layout),len(map_layout[0]))

        #Unnormalise Image for display
        clean_img = torchvision.io.read_image(f"./data/imgs/test/{label}/{global_index}.png").permute(1,2,0)
        clean_img = np.array(clean_img)

        #For individual ranking stats
        if track_stats:
            stat2 = track_stats_2d(grad_max, (5,9), map_layout, row)
            square_stats.append(stat2)
            

        # # Create a subplot to display the original image and saliency map side by side
        if plot or save_plot:
            # images = np.hstack((cam , cam_image))
            # grad_cam = Image.fromarray(images)
            actions_list = ["INT","Right","Left","Down","Up",]
            plt.close()
            plt.figure()

            # Create a subplot to display the original image and saliency map side by side
            plt.rcParams["figure.figsize"] = (10, 5)
            plt.subplot(1,2,1)
            plt.imshow(grad_max)
            plt.title(f"Saliency (Max)\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.axis('off')

            #Original Image
            plt.subplot(1,2,2)
            plt.title(f"Original (Blue)\nPred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {global_index}")
            plt.imshow((clean_img))
            plt.axis('off')

            plt.tight_layout()

            if plot:
                plt.show()
                print("Probabilities:",torch.nn.functional.softmax(scores))

            if save_plot:
                plt.savefig(fname=f"./results/{subdir}/{global_index}.png")


    return square_stats

def saliency_MLP(testset, model, subdir="v1", index_list = None, plot = False, save_plot=False):
    #Warning - validitiy not ensured due to different input types
    print("WARNING - validity/relevance not ensured due to inconsistent input types")
    label_list = ["P2_X", "P2_Y", "P1_O1","P1_O2","P1_Onion","P1_Dish", "P1_Soup", "P2_O1","P2_O2","P2_Onion","P2_Dish", "P2_Soup","TP_X","TP_Y","TO_X","TO_Y","TD_X","TD_Y","TS_X","TS_Y"]
    grad_abs_list = []
    model.to(device) #ensures consistent device used

    if index_list is None:
        index_list = range(len(testset)) #If none specified, iterate over entire dataset

    for idx in index_list:
        item = testset.__getitem__(idx)
        image_original = item[0]
        label = item[1]

        # Add a batch dimension to the input image (via .unsqueeze()) and set requires_grad to True for saliency analysis
        image = image_original.unsqueeze(0).to(device)
        image.requires_grad = True

        # Compute the scores and gradients for the input image
        # To compute the scores, do a forward pass of the image and then take the argmax
        # Use this index to extract the score_max value from "scores"
        # Then perform a backward step so that it backpropagates the gradient
        model.eval()
        scores = model.forward(image)
        score_max_index = torch.argmax(scores)
        pred = score_max_index
        score_max = scores[0, score_max_index]
        score_max.backward() # Do the backward step here for the gradient calculation

        # Calculate the saliency map by finding the maximum absolute gradient values across channels
        # You can use .abs() and torch.max()
        grad = image.grad
        grad_abs = torch.squeeze(grad.abs()).cpu()

        # # Create a subplot to display the original image and saliency map side by side
        if plot or save_plot:
            actions_list = ["INT","Right","Left","Down","Up"]
            plt.close()
            plt.figure()
            plt.rcParams["figure.figsize"] = (10, 4)

            #Saliency Bar Graph
            plt.subplot(1,2,1)
            plt.bar(range(1,len(grad_abs)+1),grad_abs)
            plt.title(f"Saliency Analysis\n Pred: {actions_list[pred]}, GT: {actions_list[label]}, Index: {idx}")
            plt.xticks(ticks = range(1,len(grad_abs)+1),labels=label_list,rotation='vertical')

            #Original Image
            plt.subplot(1,2,2)
            image_disp = torchvision.io.read_image(f"./data/imgs/test/{label}/{idx}.png")
            image_disp_rgb = image_disp.permute(1,2,0)

            plt.title(f"Map: {idx}")
            plt.axis('off')
            plt.imshow(image_disp_rgb)
            plt.tight_layout()

            if plot:
                plt.show()

            if save_plot:
                plt.savefig(fname=f"./results/{subdir}/{idx}.png")
            
        grad_abs_list.append(np.array(grad_abs))

    return grad_abs_list



















