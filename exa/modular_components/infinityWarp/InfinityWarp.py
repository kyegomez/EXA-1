import torch.multiprocessing as mp
import time 
import copy 


class InfinityWarp:
    def __init__(self, model, train_data, train_labels, infer_data, train_fn, infer_fn):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.infer_data = infer_data
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def train_model(self):
        self.train_fn(self.model, self.train_data, self.train_labels)

    def perform_inference(self):
        while True:
            #perform a deep copy of the model parameters to avoid any conflict
            model_copy = copy.deepcopy(self.model)
            preds = self.infer_fn(model_copy, self.infer_data)
            print(f"Inference result: {preds}")

            time.sleep(0.5)

    def start(self):
        train_process = mp.Processor(target=self.train_model)
        infer_process = mp.Processor(target=self.perform_inference)

        train_process.start()
        infer_process.start()

        train_process.join()
        infer_process.join()
        
