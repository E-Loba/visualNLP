import torch

from random import shuffle
# from tensorflow import Tensor as tf_t
from transformers import T5Config
from transformers import T5TokenizerFast
from keras_model import make_datasets, get_cnn_model
from ImageEncoder import T5ForImageCaptioning
from transformers import AdamW
from os.path import join as join_path
from datetime import datetime


def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    #valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch']  #, valid_loss_min.item()


def do_things():
    my_tok = T5TokenizerFast.from_pretrained("t5-small")
    # print(my_tok)
    # print(len(my_tok))
    # print("\n")
    train, valid, augm = make_datasets(raw=False, my_tok=my_tok, mx_len=MAX_LENGTH, batch_size=BATCH_SIZE)
    img_model = get_cnn_model()

    t5_config = T5Config(
        vocab_size=len(my_tok),
        decoder_start_token_id=0,
    )
    t5_model = T5ForImageCaptioning(t5_config, img_dim=IMAGE_DIM)

    optim = AdamW(t5_model.parameters(), lr=5e-5)
    device_ = torch.device('cuda')
    t5_model.to(device_)
    t5_model.train()
    data_len = 0

    start_time = datetime.now()
    for epoch_ in range(EPOCHS):
        if not data_len == 0:
            train.shuffle(data_len)
            data_len = 0
        epoch_time = datetime.now()
        for batch_img, batch_seq in train:
            # print(batch_img)
            # print("\n")
            # print(batch_seq)
            data_len += 1
            batch_img = augm(batch_img)
            img_embed = img_model(batch_img)
            captions = list(range(CAPTIONS_PER_IMAGE))
            shuffle(captions)
            for cap_ in captions:
                optim.zero_grad()
                batch_cap = batch_seq[:,cap_, :]
                batch_cap = batch_cap.numpy()
                outs = t5_model(
                        input_ids=torch.tensor(img_embed.numpy(), device=device_),
                        decoder_input_ids=torch.tensor(batch_cap[:, :-1], device=device_),
                        labels=torch.tensor(batch_cap[:, 1:], device=device_, dtype=torch.long)
                        )
                loss = outs[0]
                loss.backward()
                optim.step()
                now = datetime.now()
                print(f"epoch: {epoch_ + 1:>3}  |  step: {data_len:>5}  |  epoch time: {now - epoch_time}  |  total time: {now - start_time}  |  loss: {loss}")
        current_ckp = {
            "epoch": epoch_ + 1,
            "stat_dict": t5_model.state_dict(),
            "optimizer": optim.state_dict()
        }
        save_ckp(current_ckp, join_path(CKP_DIR, "ckp.pth"))
    t5_model.eval()
    return


CAPTIONS_PER_IMAGE = 5
MAX_LENGTH = 25
BATCH_SIZE = 12
IMAGE_DIM = 1280
EPOCHS = 4
CKP_DIR = "checkpoints"
if __name__ == "__main__":
    # my_tok = T5TokenizerFast.from_pretrained("t5-small")
    do_things()
