import multiprocessing 
import time
import copy 
import torch 


class ConcurrentTrainInference:
    def __init__(self, model, train_data, train_labels, infer_data):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.infer_data = infer_data
    
    def train_model(self, lock, model_params):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for data, labels in zip(self.train_data, self.train_labels):
            with lock:
                self.model_train_step(data, labels)
                torch.save(self.model.state_dict(), model_params)
            time.sleep(0.1) # simulate training time
    
    def perform_inference(self, lock, model_params):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        while True:
            with lock:
                model_copy = copy.deepcopy(self.model)
                model_copy.load_state_dict(torch.load(model_params))
            preds = model_copy.infer(self.infer_data)
            print(f"Inference results {preds}")
            time.sleep(0.5)

    def start(self):
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        model_params = manager.Value(str, "")

        train_process = multiprocessing.Process(target=self.train_model, args=(lock, model_params))
        infer_process = multiprocessing.Process(target=self.perform_inference, args=(lock, model_params))

        train_process.start()
        infer_process.start()

        train_process.join()
        infer_process.terminate()
        
