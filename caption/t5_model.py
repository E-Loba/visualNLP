import torch

from random import shuffle
from transformers.models.t5.configuration_t5 import T5Config
from transformers.utils.dummy_tokenizers_objects import T5TokenizerFast
from keras_model import make_datasets, get_cnn_model, decode_and_resize
from ImageEncoder import T5ForImageCaptioning
from transformers import AdamW


def process_func(img_, capt_):
    return decode_and_resize(img_), my_tok(capt_, padding="max_length", max_length=MAX_LENGTH)


def do_things():
    train, valid, augm = make_datasets(raw=False)
    img_model = get_cnn_model()

    t5_config = T5Config(
        vocab_size=len(my_tok),
        d_model=768,
        decoder_start_token_id=0,

    )
    t5_model = T5ForImageCaptioning(t5_config)

    optim = AdamW(t5_model.parameters(), lr=5e-5)
    device_ = torch.device('cuda')
    t5_model.to(device_)
    t5_model.train()

    for batch_img, batch_seq in train:
        # print(batch_img)
        # print("\n")
        # print(batch_seq)
        batch_img = augm(batch_img)
        img_embed = img_model(batch_img)
        captions = list(range(CAPTIONS_PER_IMAGE))
        shuffle(captions)
        for cap_ in captions:
            optim.zero_grad()
            batch_cap = batch_seq[:,cap_, :]
            outs = t5_model(
                    input_ids=torch.tensor(img_embed, device=device_),
                    decoder_input_ids=torch.tensor(batch_cap[:, :-1], device=device_),
                    labels=torch.tensor(batch_cap[:, 1:], device=device_)
                    )
            loss = outs[0]
            loss.backward()
            optim.step()
        break
    t5_model.eval()
    return


CAPTIONS_PER_IMAGE = 5
MAX_LENGTH = 25
if __name__ == "__main__":
    my_tok = T5TokenizerFast.from_pretrained("t5-small")
    do_things()
