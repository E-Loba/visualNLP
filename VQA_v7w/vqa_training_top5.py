#import numpy
import numpy
import torch
import csv
# import torchvision
# import shutil

from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from random import sample, randint
from datetime import datetime
from numpy import argmax
#from typing import Dict
from os.path import join, isfile
#from PIL import Image
# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.utils.dummy_sentencepiece_objects import T5Tokenizer
# from transformers.utils.dummy_tokenizers_objects import PreTrainedTokenizerFast
# from image_encode_files import SiT_base, requires_grad, create_model, distortImages
from T5v7w import T5v7w
from transformers import T5TokenizerFast, T5Config, AdamW
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter


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


def do_things(do_training:bool, image_data_path="", question_data_path="", log_dir="", ckpt_path="", num_shards=1, num_epochs=1):
    tokeniser = T5TokenizerFast.from_pretrained("t5-small")
    #tokeniser = PreTrainedTokenizer()
    tokeniser.add_special_tokens(
        {
            "additional_special_tokens": ["[TRUE]", "[FALSE]"]
        }
    )
    print()
    print("created tokeniser\n")

    config = T5Config(
        vocab_size=len(tokeniser),
        d_model=768,
        decoder_start_token_id=0
    )
    
    model = T5v7w(config)

    print()
    print("created image encoder\n")

    train_dataset = Dataset.load_from_disk(question_data_path)
    print()
    print("processed dataset\n")

    device = torch.device('cuda')

    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)

    if isfile(ckpt_path):
        load_ckp(ckpt_path, model, optim)
        print(f"loaded model from {ckpt_path}")
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)

    if num_shards:
        train_dataset = train_dataset.shard(num_shards, randint(0,num_shards-1))
    if do_training:
        print("starting training")
        writer = SummaryWriter(log_dir=log_dir)
        start_time = datetime.now()
        for epoch in range(num_epochs):
            counter = 0
            loss_avg = 0
            epoch_loss = 0
            print(f"\nstarting epoch {epoch + 1}")
            epoch_time = datetime.now()
            for item in train_dataset.shuffle(seed=epoch_time.microsecond):  #.select(list(range(10)))
                texts = item["labels"]
                # texts format
                # (labels, decoder_input, encoder_input, decoder attention mask, encoder attention mask)
                for data_item in sample(texts, len(texts)):
                    label, decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = data_item
                    # encoder_input = encoder_input[0]
                    optim.zero_grad()
                    try:
                        inputs_embeds = model.shared(torch.as_tensor(encoder_input, dtype=torch.long).to(device))
                        hidden_states = torch.cat((torch.as_tensor(item["encoder_output"]).to(device), inputs_embeds), 1)

                        outputs = model(
                            input_ids=None,
                            attention_mask=torch.as_tensor(encoder_attn_mask, dtype=torch.uint8).to(device) if encoder_attn_mask else None,
                            inputs_embeds=hidden_states,
                            decoder_input_ids=torch.as_tensor(decoder_input, dtype=torch.long).to(device) if decoder_input else None,
                            decoder_attention_mask=torch.as_tensor(decoder_attn_mask, dtype=torch.uint8).to(device) if decoder_attn_mask else None,
                            encoder_outputs=None,  # tuple( (batch size, seq len, hidden dim) , _ , _ )
                            labels=torch.as_tensor(label, dtype=torch.long).to(device)
                            )
                        # if counter == 0:
                        #     print(torch.as_tensor(item["encoder_output"]).shape)
                        #     print(torch.as_tensor(label).shape)
                    except AttributeError as E:
                        print("encoder output shape")
                        print(torch.tensor(item["encoder_output"]).shape)
                        print()
                        print("hidden states")
                        print(hidden_states)
                        print(hidden_states.shape)
                        print()
                        for entry in data_item:
                            print(type(entry))
                            if entry:
                                print(torch.tensor(entry).shape)
                            print()
                        raise E
                    loss = outputs[0]
                    epoch_loss += loss
                    loss.backward()
                    optim.step()
                    loss_avg += loss
                    counter += 1
                    if counter%100 == 0:
                        now = datetime.now()
                        writer.add_scalar("Loss/train", loss_avg / 100, counter * epoch)
                        print("epoch{:>3} at step{:>7} with loss {:.8f}\tcurrent epoch: {}\ttotal time: {}".format(epoch, counter, loss_avg / 100, now - epoch_time, now - start_time))
                        loss_avg = 0
                    if counter%5000 == 0:
                        print("question:")
                        if encoder_input is None:
                            print(tokeniser.decode([int(i) for i in decoder_input[0]]))
                        elif decoder_input is None:
                            print(tokeniser.decode([int(i) for i in encoder_input[0]]))
                        else:
                            print(tokeniser.decode([int(i) for i in label[0]]))
                        logits = outputs[1].detach().cpu().numpy()[0]
                        #print(logits.shape)
                        tokens = argmax(logits, -1)
                        #print(tokens.shape)
                        print(tokeniser.decode(tokens))
                    if counter%10000 == 0:
                        checkpoint = {
                            'epoch': epoch + 1,
                            #'valid_loss_min': valid_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optim.state_dict(),
                        }
                        save_ckp(checkpoint, join(log_dir, "ckpt.pth"))
            checkpoint = {
                'epoch': epoch + 1,
                #'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
            }
            save_ckp(checkpoint, join(log_dir, "ckpt.pth"))
            now = datetime.now()
            print("Finished epoch{:>3} with loss {:.8f} in {} hours.\tTotal session time: {}\n".format(epoch, epoch_loss / counter, now - epoch_time, now - start_time))
            writer.add_scalar("Loss/train", loss_avg / 100, counter * epoch + 1)
            loss_avg = 0
            print("question:")
            if encoder_input is None:
                print(tokeniser.decode([int(i) for i in decoder_input[0]]))
            elif decoder_input is None:
                print(tokeniser.decode([int(i) for i in encoder_input[0]]))
            else:
                print(tokeniser.decode([int(i) for i in label[0]]))
            logits = outputs[1].detach().cpu().numpy()[0]
            #print(logits.shape)
            tokens = argmax(logits, -1)
            #print(tokens.shape)
            print(tokeniser.decode(tokens))
            print("\n")
        writer.flush()
        writer.close()
        model.eval()

    print("starting evaluation")
    results = []
    ground_truths = []
    outputs_ = []
    for item in train_dataset:
        for qa_pair in item["labels"]:
            label, decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = qa_pair
            
            if decoder_input is not None and encoder_input is None:
                question = decoder_input[0]
            else:
                question = encoder_input[0]
            decoder_input = torch.tensor([[0]])  # tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
            #print(decoder_input.shape)

            inputs_embeds = model.shared(torch.as_tensor(encoder_input, dtype=torch.long).to(device))
            hidden_states = torch.cat((torch.as_tensor(item["encoder_output"]).to(device), inputs_embeds), 1)
            #print(hidden_states.shape)

            temp = {"question": None, "label": None, "output": None, "label_binary": None, "output_binary": None}
            question_ = tokeniser.decode([int(tok) for tok in question if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
            temp["question"] = question_
            print(f"question:\t{question_}")
            label_ = tokeniser.decode([int(tok) for tok in label[0] if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
            temp["label"] = label_
            temp["label_binary"] = "[TRUE]" in label_
            ground_truths.append(label_)
            print(f"ground truth:\t{label_}")
            printables = "\t"
            for i in range(5):
                outputs = model.generate(
                    input_ids=None,
                    attention_mask=torch.as_tensor(encoder_attn_mask, dtype=torch.uint8).to(device),
                    decoder_input_ids=decoder_input.to(device),
                    inputs_embeds=hidden_states
                )
                outputs = outputs.detach().to("cpu").numpy()[0]
                # output_tokens = numpy.argmax(logits, -1)
                # print(outputs)
                # print(outputs.shape)
                # print(f"output {i+1}")
                output_ = tokeniser.decode([tok for tok in outputs if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
                printables = printables + "(" + str(i+1) + ") " + output_ + "\t"
            print(f"outputs:{printables}\n")
            temp["output"] = output_
            temp["output_binary"] = "[TRUE]" in output_
            outputs_.append(output_)
            results.append(temp)
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("accuracy")
    print(accuracy_score(ground_truths, outputs_))
    print("matthews coefficient")
    print(matthews_corrcoef(ground_truths, outputs_))
    try:
        print("confusion matrix")
        print(numpy.matrix(confusion_matrix(ground_truths, outputs_, labels=["[TRUE]", "[FALSE]"])))
    except ValueError:
        print("confusion matrix not applicable")
    with open(f"{log_dir}results.csv", mode="w", encoding="utf-8") as csv_file:
        d_writer = csv.DictWriter(f=csv_file,
                                  fieldnames=["question", "label", "output", "label_binary", "output_binary"],
                                  delimiter=",",
                                  quotechar='"',
                                  lineterminator="\n")
        d_writer.writeheader()
        d_writer.writerows(results)
    return


if __name__ == "__main__":
    do_things(
        do_training=False,
        image_data_path="/home/egor/Documents/python_codes/visual7w/visual7w_images/images",
        question_data_path="/home/egor/Documents/python_codes/visual7w/datasets/v7w/free_form-1_train",  #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/mcq0_train",
        log_dir="/home/egor/Documents/python_codes/visual7w/results/free_form_cont/",
        ckpt_path="/home/egor/Documents/python_codes/visual7w/results/free_form_cont/ckpt.pth",
        num_shards=10,
        num_epochs=40
        )

