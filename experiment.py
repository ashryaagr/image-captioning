import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from nltk.tokenize import  word_tokenize


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        print(config_data)
        print(name)
        print('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
       

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__max_length = config_data['generation']['max_length']
        self.__is_det = config_data['generation']['deterministic']
        self.__temp = config_data['generation']['temperature']
        self.__bs = config_data['dataset']['batch_size']
        
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__lr = config_data['experiment']['learning_rate']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = eval(config_data['experiment']['criterion'])
        self.__optimizer = torch.optim.AdamW(self.__model.parameters(), lr = self.__lr)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.device = torch.device('cpu') # BRUHHHHH
#         self.run() ## WE REMOVED IT TO MAIN TO INCLUDE TEST AS WELL
        

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)
            
#             #creating train loss txt
#             file_name = "training_losses.txt"
#             file_path = os.path.join(self.__experiment_dir, file_name)
#             with open(file_path, "w") as f:
#                 f.write(str([]))
                
#             #creating val loss txt
#             file_name = "val_losses.txt"
#             file_path = os.path.join(self.__experiment_dir, file_name)
#             with open(file_path, "w") as f:
#                 f.write(str([]))

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()
        

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()

            # TODO
            val_loss = self.__val() 

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        self.__optimizer.zero_grad()
        # Iterate over the data, implement the training function
        for i, (images, captions, lengths) in enumerate(self.__train_loader):

            iter_loss = 0
            images = images.to(self.device)
            captions = captions.to(self.device)
          
            out = self.__model(images, captions[:,:-1]) ## Removing EOS token
        

            # We can go ahead with the following structure:
            # Model should have 2 forward functions: one for training which simply outputs a tensor
            # and one for testing which outputs a list of strings.
            
            ## Gotcha!!

            # TODO: Depending on loss criterion chosen, another architecture choice we need to be sure of is:
            # Will the model output a tensor of shape:  BATCH, SEQ_LEN, VOCAB -- reconfirm this
            
            ## Yup the shape would be BATCH, SEQ_LEN, VOCAB

            # TODO: Might need to re-check this
            ## Had to reshape it to match torch's requirement
            #print(f"pre loss=> Out shape: {out.reshape(-1,out.shape[2]).shape}, captions shape: {captions.reshape(-1).shape}")

            loss = self.__criterion(out.reshape(-1,out.shape[2]), captions.reshape(-1)) #try view LATER
            
            self.__optimizer.zero_grad()
            loss.backward(loss)
            self.__optimizer.step()
            
            iter_loss = loss.item()
            training_loss += loss.item()
            
          
            if i % 10 == 0:
                summary_str = f"Training => Epoch: {self.__current_epoch + 1}, iter: {i+1}, Loss: {iter_loss}"
                print(summary_str)
            
            if i % self.__bs == 0 :
                print("Gen Cap", self.__model.caption_images(images[0].unsqueeze(0), self.__vocab.idx2word, self.__max_length, self.__is_det, self.__temp))
                
                      
            iter_loss = 0
            
        training_loss = training_loss / len(self.__train_loader)
        
        
        ## Generating Captions after every epoch
        
        

        return training_loss

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                iter_loss = 0
                images = images.to(self.device)
                captions = captions.to(self.device)
                out = self.__model(images, captions[:,:-1])
                loss = self.__criterion(out.reshape(-1,out.shape[2]), captions.reshape(-1)) #try view LATER

                iter_loss = loss.item()
                val_loss += loss.item()
                
                if i % 10 == 0:
                    summary_str = f"val => Epoch: {self.__current_epoch + 1}, iter: {i+1}, Loss: {iter_loss}"
                    print(summary_str)
            
                if i % self.__bs == 0 :
                    print("Gen Cap val", self.__model.caption_images(images[0].unsqueeze(0), self.__vocab.idx2word, self.__max_length, self.__is_det, self.__temp))
                    
                iter_loss = 0
            
        val_loss /= len(self.__val_loader)

        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0
        cnt = 0

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                iter_loss = 0
                images = images.to(self.device)
                captions = captions.to(self.device)
                out = self.__model(images, captions[:,:-1])
                loss = self.__criterion(out.reshape(-1,out.shape[2]), captions.reshape(-1)) #try view LATER

                iter_loss = loss.item()
                test_loss += loss.item()
                
                if i % 10 == 0:
                    summary_str = f"test => Epoch: {self.__current_epoch + 1}, iter: {i + 1}, Loss: {iter_loss}"
                    print(summary_str)
                    
#                 print(len(img_ids))
#                 print(img_ids)
                
                for j, image in enumerate(images):
                    cnt += 1 # increase counter
                    
                    pred_caption = self.__model.caption_images(image.unsqueeze(0), self.__vocab.idx2word, self.__max_length, self.__is_det, self.__temp)
                    
                    # Removing start, pad, end tokens and tokenizing & lowering
                    try:
                        endIndex = pred_caption.index("<end>")
                        pred_caption = pred_caption[1:endIndex]
                    except:
                        pred_caption = pred_caption[1:]
                    pred_caption = " ".join(pred_caption)
                    pred_caption = word_tokenize(pred_caption.lower())
                    
                    targetCaption = []

                    for anns in self.__coco_test.imgToAnns[img_ids[j]]: # Get alltrue captions
#                         print(anns) has 'image_id', 'id', 'caption'
#                         sys.exit("bruh")
                        cap = anns['caption']
                        cap = str(cap).lower()
                        cap = word_tokenize(cap)
                        targetCaption.append(cap)
                    
                
                    bleu1_score += bleu1(targetCaption, pred_caption)
                    bleu4_score += bleu4(targetCaption, pred_caption)
                    
                    
#                     targetCaption = [self.__vocab.idx2word[key.item()] for key in captions[j]]
                    
#                     print("isitwotking",captions[j], pred_caption)
                    if cnt % 1000 == 0:
                        print("Predicted: "," ".join(pred_caption),"\n")
                        print("Targets: ")
                        for sent in targetCaption:
                            print(" ".join(sent))
                        print("\n\n")
                        
                
           
                iter_loss = 0
            
            
        test_loss /= len(self.__test_loader)
        bleu1_score /= cnt
        bleu4_score /= cnt
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,bleu1_score,bleu4_score)
       
        self.__log(result_str)

        return test_loss, bleu1_score, bleu4_score

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        #plt.show()

        
    def test1image(self):
        self.__model.eval()
        rand = np.random.randint(0,self.__bs)
        
        x ,y , img_id = next(iter(self.__test_loader))
        x = x.to(self.device)[rand]
        y = y.to(self.device)[rand]
        img_id = img_id[rand]
        
        
        pred_caption = self.__model.caption_images(x.unsqueeze(0), self.__vocab.idx2word, max_length=self.__max_length, is_deterministic=True, temp=self.__temp)
        try:
            endIndex = pred_caption.index("<end>")
            pred_caption = pred_caption[1:endIndex]
        except:
            pred_caption = pred_caption[1:]
        pred_caption = " ".join(pred_caption)
        print("pred caption", pred_caption)
        
        pred_caption = word_tokenize(pred_caption.lower())
        

        
        targetCaption = []
        for anns in self.__coco_test.imgToAnns[img_id]: # Get alltrue captions
            cap = anns['caption']
            cap = str(cap).lower()
            cap = word_tokenize(cap)
            targetCaption.append(cap)
#         targetCaption = [self.__vocab.idx2word[key.item()] for key in y]
        print("\nTargets:")
        for sent in targetCaption:
            print(" ".join(sent))
        
        print(f"Bleu1 {bleu1(targetCaption, pred_caption)}, Bleu4 {bleu4(targetCaption, pred_caption)}")
        
        plt.imshow(x.cpu().permute(1,2,0).numpy())
        plt.savefig("testImages/testImg.png")
#         plt.show()

    def extractCaption_helper(self, pred_caption):
        try:
            endIndex = pred_caption.index("<end>")
            pred_caption = pred_caption[1:endIndex]
        except:
            pred_caption = pred_caption[1:]
        pred_caption = " ".join(pred_caption)

        return pred_caption
        
    def generateReportCaptions_helper(self, good=True, cnt=1):
        self.__model.eval()
        prefix = "good" if good else "bad"

        for i in range(1):
            if cnt==0:
                return

            rand = np.random.randint(0,self.__bs)
            
            x ,y , img_id = next(iter(self.__test_loader))
            x = x.to(self.device)[rand]
            y = y.to(self.device)[rand]
            img_id = img_id[rand]

            def log_for_img_id(log_str, newLine=False):
                log_to_file_in_dir(self.__experiment_dir, f"{prefix}_{img_id}.txt", log_str)
                if newLine:
                    log_to_file_in_dir(self.__experiment_dir, f"{prefix}_{img_id}.txt", "")


            targetCaption = []
            for anns in self.__coco_test.imgToAnns[img_id]: # Get alltrue captions
                cap = anns['caption']
                cap = str(cap).lower()
                cap = word_tokenize(cap)
                targetCaption.append(cap)
            
            
            pred_caption = self.__model.caption_images(x.unsqueeze(0), self.__vocab.idx2word, max_length=self.__max_length, is_deterministic=False, temp=0.4)
            pred_caption = self.extractCaption_helper(pred_caption)
            b1 = bleu1(targetCaption, word_tokenize(pred_caption.lower()))


            if good and b1>80:
                cnt -= 1
                pass
            elif not good and b1<40:
                cnt -= 1
                pass
            else:
                continue
            
            log_for_img_id("Model Name: "+self.__name, newLine=True)
            log_for_img_id("Targets:")
            for cap in targetCaption:
                log_for_img_id(" ".join(cap))
            log_for_img_id("")

            log_for_img_id("Temp 0.4 => "+pred_caption, newLine=True)

            pred_caption = self.__model.caption_images(x.unsqueeze(0), self.__vocab.idx2word, max_length=self.__max_length, is_deterministic=True, temp=0.4)
            pred_caption = self.extractCaption_helper(pred_caption)
            log_for_img_id("Deterministic Temp 0.4 => "+pred_caption, newLine=True)

            pred_caption = self.__model.caption_images(x.unsqueeze(0), self.__vocab.idx2word, max_length=self.__max_length, is_deterministic=False, temp=5)
            pred_caption = self.extractCaption_helper(pred_caption)
            log_for_img_id("Temp 5 => "+pred_caption, newLine=True)

            pred_caption = self.__model.caption_images(x.unsqueeze(0), self.__vocab.idx2word, max_length=self.__max_length, is_deterministic=False, temp=0.001)
            pred_caption = self.extractCaption_helper(pred_caption)
            log_for_img_id("Temp 0.001 => "+pred_caption)
            

            plt.imshow(x.cpu().permute(1,2,0).numpy())
            plt.savefig(os.path.join(self.__experiment_dir, "captions", f"{prefix}_{img_id}.png"))

        log_to_file_in_dir(self.__experiment_dir, f"report_caption.txt", f"Couldn't search for {cnt} {prefix} example.")

    def generateReportCaptions(self):
        os.makedirs(self.__experiment_dir+"/captions", exist_ok=True)
        self.generateReportCaptions_helper(good=True, cnt=6)
        self.generateReportCaptions_helper(good=False, cnt=6)
