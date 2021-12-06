#import numpy
import numpy
import torch
import csv
# import torchvision
# import shutil

from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, f1_score
from random import sample, randint
from datetime import datetime
from numpy import argmax
#from typing import Dict
from os.path import join, isfile

# from transformers.utils.dummy_pt_objects import BertForMaskedLM
#from PIL import Image
# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.utils.dummy_sentencepiece_objects import T5Tokenizer
# from transformers.utils.dummy_tokenizers_objects import PreTrainedTokenizerFast
# from image_encode_files import SiT_base, requires_grad, create_model, distortImages
from T5v7w import T5v7w
from transformers import T5ForConditionalGeneration
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


def do_things(
                do_encoder=False,
                do_training=True,
                question_data_path="",
                log_dir="",
                ckpt_path="",
                num_shards=1,
                num_epochs=1,
                is_boxes=False,
                shard_=None,
                do_eval=True,
            ):
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
    
    if do_encoder:
        model = T5ForConditionalGeneration(config)
    else:
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
    
    if do_training:
        all_steps = 0
        if num_shards:
            if not shard_:
                shard_ = randint(0,num_shards-1)
            train_dataset = train_dataset.shard(num_shards, shard_)
        print("starting training")
        writer = SummaryWriter(log_dir=log_dir)
        epoch_writer = SummaryWriter(log_dir=join(log_dir, "epochs"))
        start_time = datetime.now()
        for epoch in range(num_epochs):
            counter = 0
            loss_avg = 0
            avg_counter = 0
            epoch_loss = 0
            print(f"\nstarting epoch {epoch + 1}")
            epoch_time = datetime.now()
            for item in train_dataset.shuffle(seed=epoch_time.microsecond):  #.select(list(range(10)))
                texts = item["labels"]
                if texts is None:
                    continue
                # texts format
                # (labels, decoder_input, encoder_input, decoder attention mask, encoder attention mask)
                for data_item in sample(texts, len(texts)):
                    if is_boxes:
                        label = [data_item]
                        decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = None, None, None, None
                    else:
                        label, decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = data_item
                    # encoder_input = encoder_input[0]
                    optim.zero_grad()
                    try:
                        if is_boxes:
                            hidden_states = torch.as_tensor(item["encoder_output"]).to(device)
                        else:
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
                    avg_counter += 1
                    counter += 1
                    all_steps += 1
                    if counter%100 == 0:
                        now = datetime.now()
                        writer.add_scalar("Loss/train", loss_avg / avg_counter, all_steps)
                        print("epoch{:>3} at step{:>7} with loss {:.8f}\tcurrent epoch: {}\ttotal time: {}".format(epoch, counter, loss_avg / avg_counter, now - epoch_time, now - start_time))
                        loss_avg = 0
                        avg_counter = 0
                    if counter%5000 == 0:
                        print("question:")
                        if decoder_input is not None:
                            print(tokeniser.decode([int(i) for i in decoder_input[0]]))
                        elif encoder_input is not None:
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
            print("Finished epoch{:>3} with loss {:.8f} in {} hours.\tTotal session time: {}\n".format(epoch+1, epoch_loss / counter, now - epoch_time, now - start_time))
            writer.add_scalar("Loss/train", loss_avg / avg_counter, all_steps)
            epoch_writer.add_scalar("Loss/epoch", epoch_loss / counter, epoch)
            loss_avg = 0
            avg_counter = 0
            print("question:")
            if decoder_input is not None:
                print(tokeniser.decode([int(i) for i in decoder_input[0]]))
            elif encoder_input is not None:
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

    if do_eval:
        print("starting evaluation")
        model.eval()
        results = []
        ground_truths = []
        outputs_ = []
        #with torch.no_grad():
        for item in train_dataset:
            if not item["labels"]:
                continue
            for qa_pair in item["labels"]:
                if is_boxes:
                    label = [data_item]
                    decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = None, None, None, None
                else:
                    label, decoder_input, encoder_input, decoder_attn_mask, encoder_attn_mask = qa_pair
                
                if decoder_input is not None:
                    question = decoder_input[0]
                elif encoder_input is not None:
                    question = encoder_input[0]
                else:
                    question = ""
                decoder_input = torch.tensor([[0]])  # tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
                #print(decoder_input.shape)

                if is_boxes:
                    hidden_states = torch.as_tensor(item["encoder_output"]).to(device)
                else:
                    inputs_embeds = model.shared(torch.as_tensor(encoder_input, dtype=torch.long).to(device))
                    hidden_states = torch.cat((torch.as_tensor(item["encoder_output"]).to(device), inputs_embeds), 1)
                #print(hidden_states.shape)

                temp = {"question": None, "label": None, "output": None, "label_binary": None, "output_binary": None}
                question_ = tokeniser.decode([int(tok) for tok in question if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
                temp["question"] = question_
                img_id = item["image_id"]
                print(f"image {img_id} with question:\t{question_}")
                label_ = tokeniser.decode([int(tok) for tok in label[0] if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
                temp["label"] = label_
                temp["label_binary"] = "[TRUE]" in label_
                ground_truths.append(label_)
                print(f"ground truth:\t{label_}")
                printables = "\t"
                if encoder_attn_mask is not None:
                    make_mask = torch.as_tensor(encoder_attn_mask, dtype=torch.uint8).to(device)
                else:
                    make_mask = torch.ones(hidden_states.size()[:-1]).to(device)
                outputs = model.generate(
                    input_ids=None,
                    attention_mask=make_mask,
                    decoder_input_ids=decoder_input.to(device),
                    inputs_embeds=hidden_states,
                    # do_sample=True,
                    # top_p=0.92,
                    # top_k=50,
                    num_beams=10,
                    num_beam_groups=5,
                    diversity_penalty=0.8,
                    num_return_sequences=5
                )
                top_k = []
                for i, output_ in enumerate(outputs):
                    output_ = output_.detach().to("cpu").numpy()
                    output_ = tokeniser.decode([tok for tok in output_ if tok not in [-100, 0, 1, tokeniser.pad_token_id]])
                    printables = printables + "(" + str(i+1) + ") " + output_ + "\t"
                    top_k.append(output_)
                print(f"outputs:{printables}\n")
                temp["output"] = output_
                temp["output_binary"] = "[TRUE]" in output_
                outputs_.append(top_k)
                results.append(temp)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        #print("accuracy")
        #print(accuracy_score(ground_truths, outputs_))
        #print("matthews coefficient")
        #print(matthews_corrcoef(ground_truths, outputs_))
        # try:
        #     print("confusion matrix")
        #     print(numpy.matrix(confusion_matrix(ground_truths, outputs_, labels=["[TRUE]", "[FALSE]"])))
        # except ValueError:
        #     print("confusion matrix not applicable")
        print("top_k")
        print(top_k_acc(ground_truths, outputs_))
        print("top_1")
        print(accuracy_score(ground_truths, [k[0] for k in outputs_]))
        print("mcq")
        print(mcq_acc(ground_truths, [k[0] for k in outputs_]))
        # print("matthews coefficient")
        # print(matthews_corrcoef(ground_truths, [k[0] for k in outputs_]))
        # print("f1 score")
        # print(f1_score(ground_truths, [k[0] for k in outputs_]))
        with open(f"{log_dir}results.csv", mode="w", encoding="utf-8") as csv_file:
            d_writer = csv.DictWriter(f=csv_file,
                                    fieldnames=["question", "label", "output", "label_binary", "output_binary"],
                                    delimiter=",",
                                    quotechar='"',
                                    lineterminator="\n")
            d_writer.writeheader()
            d_writer.writerows(results)
    train_dataset.cleanup_cache_files()
    return


def top_k_acc(truth, preds):
    """
    truth: num_samples
    preds: [num_samples, k]
    """
    if len(truth) != len(preds):
        raise ValueError
    acc = 0
    for t, k in zip(truth, preds):
        if any([k_ == t for k_ in k]):
            acc += 1
    return acc / len(truth)


def mcq_acc(truth, preds):
    """
    truth: list of answers
    preds: list of mcq candidate answer choices by the model
    """
    acc = 0
    for t, k in zip(truth, preds):
        if t in k:
            acc += 1
    return acc / len(truth)


if __name__ == "__main__":
    shards = 20
    # for shard_step in range(1, shards):
    #     do_things(
    #         do_training=True,
    #         image_data_path="/home/egor/Documents/python_codes/visual7w/visual7w_images/images",
    #         question_data_path="/home/egor/Documents/python_codes/visual7w/datasets/v7w/boxes_48_1",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/free_form_eval-1_val",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/boxes_48_last",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/boxes_48_1",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/free_form-1_train",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/mcq0_train",
    #         log_dir="/home/egor/Documents/python_codes/visual7w/results/pretrain",
    #         #"/home/egor/Documents/python_codes/visual7w/results/pretrain_freeform/",
    #         ckpt_path="/home/egor/Documents/python_codes/visual7w/results/pretrain/ckpt.pth",
    #         #"/home/egor/Documents/python_codes/visual7w/results/free_form_no_distort/ckpt.pth",
    #         num_shards=shards,
    #         num_epochs=15,  # finetune: 20
    #         is_boxes=True,
    #         shard_=shard_step,  # shards-1-shard_step
    #         do_eval=False
    #         )
    do_things(
            do_encoder=True,
            do_training=False,
            question_data_path="/home/egor/Documents/python_codes/visual7w/datasets/v7w/where-1_train",
            #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/where-1_train",
            #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/free_form-1_train",
            log_dir="/home/egor/Documents/python_codes/visual7w/results/encoder_where_mcq",
            ckpt_path="/home/egor/Documents/python_codes/visual7w/results/encoder_where_mcq/ckpt.pth",
            num_shards=1,
            num_epochs=10,  # finetune: 20
            is_boxes=False,
            shard_=None,  # shards-1-shard_step
            do_eval=True,
            )
    # do_things(
    #         do_encoder=True,
    #         do_training=False,
    #         question_data_path="/home/egor/Documents/python_codes/visual7w/datasets/v7w/where_mcq5_val",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/where-1_train",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/where-1_train",
    #         #"/home/egor/Documents/python_codes/visual7w/datasets/v7w/free_form_eval-1_val",
    #         log_dir="/home/egor/Documents/python_codes/visual7w/results/encoder_where_mcq/",
    #         ckpt_path="/home/egor/Documents/python_codes/visual7w/results/encoder_where_mcq/ckpt.pth",
    #         num_shards=20,
    #         num_epochs=20,  # finetune: 20
    #         is_boxes=False,
    #         shard_=None,  # shards-1-shard_step
    #         do_eval=True,
    #         )
