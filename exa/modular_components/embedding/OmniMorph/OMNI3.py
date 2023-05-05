import torch
import torch.nn as nn



class OmniMorph(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._embedding_registry = {}
        self._embedding_instances = {}

    def register_embedding(self, modality_type, embedding_class):
        self._embedding_registry[modality_type] = embedding_class


    def instantiate_embedding(self, modality_type, embedding_class=None, *args, **kwargs):
        if embedding_class is None:
            embedding_class = self._embedding_registry.get(modality_type)
        
        if embedding_class is not None:
            self._embedding_instances[modality_type] = embedding_class(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
        

    def forward(self, input_data, modality_type=None, **kwargs):
        if modality_type is None:
            modality_type = self.detect_modality(input_data)
        
        embedding_instance = self._embedding_instances.get(modality_type)
        if embedding_instance is not None:
            return embedding_instance(input_data, **kwargs)
        else:
            raise ValueError(f"Embedding for modality type {modality_type} not instantiated")\
            
    def detect_modality(self, input_data):
        if len(input_data.shape) == 2 and input_data.dtype == torch.int64:
            return 'text'
        elif len(input_data.shape) == 4:
            return 'vision'
        elif len(input_data.shape) == 3:
            return 'audio'
        elif len(input_data.shape) == 5:
            return 'video'
        else:
            raise ValueError("Unable to detect input data modality")
        


omni_morph = OmniMorph()

# Register and instantiate embeddings
omni_morph.register_embedding('text', TextEmbedding)
omni_morph.instantiate_embedding('text', num_embeddings=10000, embedding_dim=768)

omni_morph.register_embedding('vision', VisionEmbedding)
omni_morph.instantiate_embedding('vision', img_size=224, patch_size=16, in_chans=3, embed_dim=768)

omni_morph.register_embedding('audio', AudioEmbedding)
omni_morph.instantiate_embedding('audio', in_channels=128, embed_dim=768)

omni_morph.register_embedding('video', VideoEmbedding)
omni_morph.instantiate_embedding('video', num_channels=3, time_dim=10, height=224, width=224, embed_dim=768)

# Example usage for different modalities
text_input = torch.randint(0, 10000, (1, 50))
vision_input = torch.randn(1, 3, 224, 224)
audio_input = torch.randn(1, 128, 100)
video_input = torch.randn(1, 3, 10, 224, 224)

text_embedding = omni_morph(text_input)  # modality_type is automatically detected
vision_embedding = omni_morph(vision_input)  # modality_type is automatically detected
audio_embedding = omni_morph(audio_input)  # modality_type is automatically detected
video_embedding = omni_morph(video_input)  # modality_type is automatically detected
