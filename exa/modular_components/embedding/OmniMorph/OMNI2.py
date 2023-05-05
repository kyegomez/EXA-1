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


class PositionalEmbedding(nn.Embedding):
    def forward(
        self,
        x,
        positions=None,
        **kwargs,
    ):
        if positions is None:
            # being consistent with Fairseq, which starts from 2.
            positions = (
                torch.arange(2, x.size(1) + 2, device=x.device).long().unsqueeze(0)
            )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class OmniMorph(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._embedding_registry = {}
        self._embedding_instances = {}

    def register_embedding(self, modality_type, embedding_class):
        self._embedding_registry[modality_type] = embedding_class

    def instantiate_embedding(self, modality_type, *args, **kwargs):
        embedding_class = self._embedding_registry.get(modality_type)
        if embedding_class is not None:
            self._embedding_instances[modality_type] = embedding_class(*args, **kwargs)
            self.add_module(f"{modality_type}_embedding", self._embedding_instances[modality_type])
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")

    def forward(self, input_data, modality_type=None, **kwargs):
        if modality_type is None:
            modality_type = self.detect_modality(input_data)

        embedding_instance = self._embedding_instances.get(modality_type)
        if embedding_instance is not None:
            return embedding_instance(input_data, **kwargs)
        else:
            raise ValueError(f"Embedding for modality type {modality_type} not instantiated.")

    def detect_modality(self, input_data):
        # Implement heuristics to automatically detect input data modality
        # For example:
        if len(input_data.shape) == 2 and input_data.dtype == torch.int64:
            return 'text'
        elif len(input_data.shape) == 4:
            return 'vision'
        elif len(input_data.shape) == 3:
            return 'audio'
        else:
            raise ValueError("Unable to detect input data modality.")



class AudioEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    
    def forward(self, x, **kwargs):
        return self.conv(x)
    



# Instantiate OmniMorph
omni_morph = OmniMorph()

# Register and instantiate embeddings
omni_morph.register_embedding('text', TextEmbedding)
omni_morph.instantiate_embedding('text', num_embeddings=10000, embedding_dim=768)

omni_morph.register_embedding('vision', VisionEmbedding)
omni_morph.instantiate_embedding('vision', img_size=224, patch_size=16, in_chans=3, embed_dim=768)

omni_morph.register_embedding('audio', AudioEmbedding)
omni_morph.instantiate_embedding('audio', in_channels=128, embed_dim=768)

# Example usage for different modalities
text_input = torch.randint(0, 10000, (1, 50))
vision_input = torch.randn(1, 3, 224, 224)
audio_input = torch.randn(1, 128, 100)

text_embedding = omni_morph(text_input)  # modality_type is automatically detected
vision_embedding = omni_morph(vision_input)  # modality_type is automatically detected
audio_embedding = omni_morph(audio_input)  # modality_type is automatically detected