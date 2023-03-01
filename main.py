from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        
        try :
            mode = sys.argv[2]
        except:
            mode = "train"
            

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    
    if mode == "train":
        exp.run() 
    elif mode == "test":
        test_loss, bleu1_score, bleu4_score = exp.test()
        print(f"test loss: {test_loss}, bleu1: {bleu1_score}, bleu4: {bleu4_score}")
    else:
        print("invalid mode")
