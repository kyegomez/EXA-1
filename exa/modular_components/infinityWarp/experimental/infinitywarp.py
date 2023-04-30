import threading 
import time
import copy 

class ConcurrentTrainInference:
    def __init__(self, model, train_data, train_labels, infer_data):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.infer_data = infer_data
    
    def train_model(self):
        for data, labels in zip(self.train_data, self.train_labels):
            self.model.train_step(data, labels)
            time.sleep(0.1) # sumulate training time
        
    def perform_inference(self):
        while True:
            #perform a deep copy of the models params to avoid any conflicts
            model_copy = copy.deepcopy(self.model)
            preds = model_copy.infer(self.infer_data)
            print(f"Inference results: {preds}")

            time.sleep(0.5)

    def start(self):
        train_thread = threading.Thread(target=self.train_model)
        infer_thread = threading.Thread(target=self.perform_inference)

        train_thread.start()
        infer_thread.start()

        train_thread.join()
        infer_thread.join()

# to use
# # Define your neural network model here
# class MyNeuralNet:
#     # ... (as before)

# # Initialize model and training/inference data
# model = MyNeuralNet()
# train_data, train_labels = load_training_data()  # Implement data loading function
# infer_data = load_inference_data()  # Implement data loading function

# # Create and start the concurrent training and inference component
# concurrent_component = ConcurrentTrainInference(model, train_data, train_labels, infer_data)
# concurrent_component.start()
