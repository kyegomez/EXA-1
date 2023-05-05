import torch
import torch.nn as nn

class VisionLanguageEmbedding(nn.Module):
    def __init__(self, text_embed, vision_embed):
        super().__init__()
        self.text_embed = text_embed
        self.vision_embed = vision_embed

    def forward(self, textual_tokens, visual_tokens, **kwargs):
        if textual_tokens is None:
            return self.vision_embed(visual_tokens)

        if visual_tokens is None:
            return self.text_embed(textual_tokens)

        x1 = self.vision_embed(visual_tokens)
        x2 = self.text_embed(textual_tokens)

        return torch.cat([x1, x2], dim=1)


class VisionEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if contain_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()



class AudioEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    
    def forward(self, x, **kwargs):
        return self.conv(x)
    




    
class OmniMorph(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._embedding_registry = {}
        self._embedding_instances = {}
        self._fusion_techniques = {}

        # preregister and instantiate the embedding functions
        # Pre-register and instantiate the embedding functions
        
        # Pre-register and instantiate the embedding functions
        self.register_and_instantiate('text', TextEmbedding, num_embeddings=10000, embedding_dim=768)
        self.register_and_instantiate('vision', VisionEmbedding, img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        self.register_and_instantiate('audio', AudioEmbedding, in_channels=128, embed_dim=768)
        # self.register_and_instantiate('video', VideoEmbedding, num_channels=3, time_dim=10, height=224, width=224, embed_dim=768)
        
        # Instantiate VisionLanguageEmbedding with visionembeddings and textembeddings instances
        vision_embed_instance = self._embedding_instances.get('vision')
        text_embed_instance = self._embedding_instances.get('text')
        self.vision_language_embedding = VisionLanguageEmbedding(text_embed_instance, vision_embed_instance)



    def register_and_instantiate(self, modality_type, embedding_class, **kwargs):
        self.register_embedding(modality_type, embedding_class)
        self.instantiate_embedding(modality_type, **kwargs)

    def register_embedding(self, modality_type, embedding_class):
        self._embedding_registry[modality_type] = embedding_class

    def instantiate_embedding(self, modality_type, embedding_class=None, *args, **kwargs):
        if embedding_class is None:
            embedding_class = self._embedding_registry.get(modality_type)
        
        if embedding_class is not None:
            self._embedding_instances[modality_type] = embedding_class(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")

    def forward(self, input_data, modality_type=None, fusion_technique=None, **kwargs):
        if modality_type is None:
            modality_type = self.detect_modality(input_data)
            print(modality_type)
        
        embedding_instance = self._embedding_instances.get(modality_type)
        if embedding_instance is not None:
            embedding = embedding_instance(input_data, **kwargs)
            print(embedding)
            if fusion_technique:
                fusion_fn = self._fusion_techniques.get(fusion_technique)
                if fusion_fn:
                    embedding = fusion_fn(embedding)
                    print(embedding)
                else:
                    raise ValueError(f"Unsupported fusion technique: {fusion_technique}")
            return embedding
        else:
            raise ValueError(f"Embedding for modality type {modality_type} not instantiated")

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
        

    def register_fusion_technique(self, technique_name, fusion_fn):
        self._fusion_techniques[technique_name] = fusion_fn

omni_morph = OmniMorph()



text_input = torch.randint(0, 10000, (1, 50))
# vision_input = torch.randn(1, 3, 224, 224)
# audio_input = torch.randn(1, 128, 100)



# audio_input = audio_input.unsqueeze(1)  # Add a new dimension for channels

text_embedding = omni_morph(text_input)  # modality_type is automatically detected
# vision_embedding = omni_morph(vision_input)  # modality_type is automatically detected
# audio_embedding = omni_morph(audio_input)  # modality_type is automatically detected
