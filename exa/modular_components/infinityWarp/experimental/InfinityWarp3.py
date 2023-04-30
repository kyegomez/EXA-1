import torch.multiprocessing as mp
import time
import copy

class InfinityWarp:
    def __init__(self, model, train_data, train_labels, infer_data):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.infer_data = infer_data
    
    def train_model(self):
        for data, labels in zip(self.train_data, self.train_labels):
            self.model.train_step(data, labels)
            time.sleep(0.1)

    def perform_inference(self):
        while True:
            #perform a deep copy of the models params to avoid any conflict
            model_copy = copy.deepcopy(self.model)
            preds = model_copy.infer(self.infer_data)
            print(f"Inference result: {preds}")

            time.sleep(0.5)

    def start(self):
        train_process = mp.Process(target=self.train_model)
        infer_process = mp.Processor(target=self.perform_inference)

        train_process.start()
        infer_process.start()

        train_process.join()
        infer_process.join()
