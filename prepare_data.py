from random import shuffle
import torchvision
import torch

from PIL import Image
from typing import Dict
from os.path import join
from datasets import Dataset
from transformers import T5TokenizerFast
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from image_encode_files import SiT_base, requires_grad, create_model, distortImages


def image_encoder(my_size=None):
    checkpoint_path = "/home/egor/Documents/python_codes/visual7w/SiT/checkpoints/finetune/v7w2/checkpoint.pth"  # finetuned on v7w
    # "/home/egor/Documents/python_codes/visual7w/SiT/checkpoints/finetune/CIFAR10_LE/checkpoint.pth"  # raw
    # "/home/egor/Documents/python_codes/visual7w/checkpoints/checkpoint4.pth"  # pretrained
    img_size_ = image_model_size if my_size is None else my_size
    patch_size_ = 16
    model = create_model('SiT_base',
        pretrained=False,
        img_size=img_size_, patch_size=patch_size_, num_classes=0, 
        drop_rate=0.0, drop_path_rate=0.1,
        drop_block_rate=None, training_mode="SSL", representation_size=768)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # for k in ['rot_head.weight', 'rot_head.bias', 'contrastive_head.weight', 'contrastive_head.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    model.load_state_dict(checkpoint_model, strict=False)
    model.to("cpu")
    # model_ema = ModelEma(model, decay=0.99996,device='cpu', resume='')
    model.eval()
    requires_grad(model, False)
    model.rot_head.weight.requires_grad = True
    model.rot_head.bias.requires_grad = True
    model.contrastive_head.weight.requires_grad = True
    model.contrastive_head.bias.requires_grad = True
    model.pre_logits_rot.fc.weight.requires_grad = True
    model.pre_logits_rot.fc.bias.requires_grad = True
    model.pre_logits_contrastive.fc.weight.requires_grad = True
    model.pre_logits_contrastive.fc.bias.requires_grad = True

    # from SiT_pytorch.sit import SiT
    # model = SiT(image_size=224, patch_size=16, rotation_node=4, contrastive_head=512)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # print('Trainable Parameters: %.3fM' % parameters)

    t = []
    # size = int((256 / img_size_) * img_size_)
    t.append(
        torchvision.transforms.Resize(size=(img_size_, img_size_)),  # to maintain same ratio w.r.t. 224 images
    )
    #t.append(torchvision.transforms.CenterCrop(img_size_))
    t.append(torchvision.transforms.ToTensor())
    t.append(torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t_ = torchvision.transforms.Compose(t)

    return model, t_


def encode_image(model, t, image_path):
    with torch.no_grad():
        image = Image.open(image_path)
        image = image.convert("RGB")
        input_ = t(image).to("cpu")
        input_ = input_[None, :, :, :]
        # input_ = distortImages(input_)

        out_raw, out_attn = model(input_, attn=True)
    return out_attn


def concatenate(tokeniser:T5TokenizerFast, question, answer, flag, joiner=" "):
    text = joiner.join([question, answer])
    tokens = tokeniser(text, padding="max_length", max_length=198)
    labels = [tokens.input_ids]
    labels = [
           [(label if label != tokeniser.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)

    return (labels, None, None, None, None)

def tokenise_separate_in_out(tokeniser:T5TokenizerFast, question, answer, flag):
    input = tokeniser(question, return_tensors="pt", padding="max_length", max_length=198)
    label = tokeniser(answer, padding="max_length", max_length=198)
    labels = [label.input_ids]
    labels = [
           [(label if label != tokeniser.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)
    return (labels, input.input_ids, None, input.attention_mask, None)

def tokenise_separate_de_en(tokeniser, question, answer, flag):
    # text = " ".join([question, answer])
    input_ = tokeniser(question, padding="max_length", max_length=100+int((image_model_size / 16)**2), return_tensors="pt")
    #  print(input_.input_ids)

    labels = tokeniser(answer, padding="max_length" , max_length=100+int((image_model_size / 16)**2))
    labels = [labels.input_ids]
    labels = [
           [(label if label != tokeniser.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)
    attn_mask = torch.ones((1, 198), dtype=torch.float64)
    return (labels, None, input_.input_ids, None, torch.cat((attn_mask, input_.attention_mask), 1))

def concatenate_masked(tokeniser:T5TokenizerFast, question, answer, flag, joiner=" "):
    text = joiner.join([question, answer])
    tokens = tokeniser(text, padding="max_length", max_length=198)
    labels = [tokens.input_ids]
    question_length = tokeniser(question)
    question_length = len(question_length[0].ids)
    labels = [
           [(label if (label != tokeniser.pad_token_id) and (i >= question_length) else -100) for i, label in enumerate(labels_example)] for labels_example in labels
    ]
    labels = torch.tensor(labels)
    return (labels, None, None, None, None)

def mcq(tokeniser, question, answer, flag):
    text = " ".join([question, answer])
    input_ = tokeniser(text, padding="max_length", max_length=250-198, return_tensors="pt")

    labels = tokeniser("[TRUE]" if flag else "[FALSE]", padding="max_length", max_length=250)
    labels = [labels.input_ids]
    labels = [
           [(label if label != tokeniser.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)
    attn_mask = torch.ones((1, 198), dtype=torch.float64)
    return (labels, None, input_.input_ids, None, torch.cat((attn_mask, input_.attention_mask), 1))

def mcq_new(tokeniser, question, answer, flag):
    #text = " ".join([question, answer])
    input_ = tokeniser(question, padding="max_length", max_length=250-198, return_tensors="pt")

    labels = tokeniser(answer+("[TRUE]" if flag else "[FALSE]"), padding="max_length", max_length=250)
    labels = [labels.input_ids]
    labels = [
           [(label if label in tokeniser.additional_special_tokens_ids else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)
    attn_mask = torch.ones((1, 198), dtype=torch.float64)
    return (labels, None, input_.input_ids, None, torch.cat((attn_mask, input_.attention_mask), 1))


def mapping_func(encoder, img_preproc, tokeniser, text_process=concatenate, mcq=-1, q_type=None):
    r"""
    text process is a function that takes following arguments
        - a tokeniser (tokeniser)
        - the question text (str)
        - the output text (str)
        - a boolean flag indicating whether the mode is MCQ or free form
    and returns a tuple:
        - (labels, decoder_input, encoder_input, decoder_attention_mask, encoder_attention_mask)
    """
    def my_func(dict: Dict):
        if not text_process:
            raise NotImplementedError("please pass a valid text processing function")
        image_path = join(data_path, dict["filename"])
        encoder_output = encode_image(encoder, img_preproc, image_path)
        temp = []
        for pair in dict["qa_pairs"]:
            if q_type:
                if pair["type"] != q_type:
                    continue
            if mcq > len(pair["multiple_choices"]):
                candidates = [pair["answer"]] + pair["multiple_choices"]
                shuffle(candidates)
                input_ = tokeniser.eos_token.join([pair["question"]] + candidates)
                # print(input_)
                data = text_process(tokeniser, input_, pair["answer"], None)
                temp.append(data)
            else:
                data = text_process(tokeniser, pair["question"], pair["answer"], True)
                temp.append(data)
                if mcq >-1:
                    choice = pair["multiple_choices"][mcq]
                    data = text_process(tokeniser, pair["question"], choice, False)
                    temp.append(data)
        if len(temp) > 0:
            dict["encoder_output"] = encoder_output
            dict["labels"] = temp
        else:
            dict["labels"] = None
            dict["encoder_output"] = None
        return dict
    return my_func


def do_things(save_train: bool, my_func, mcq_=0, name="mcq", q_type=None):
    tokeniser = T5TokenizerFast.from_pretrained("t5-small")
    #tokeniser = PreTrainedTokenizer()
    tokeniser.add_special_tokens(
        {
            "additional_special_tokens": ["[TRUE]", "[FALSE]"]
        }
    )
    print()
    print("created tokeniser\n")

    encoder, t = image_encoder()
    train_dataset = Dataset.from_json("/home/egor/Documents/python_codes/visual7w/dataset_v7w_telling.json", field="images")
    train_dataset.add_column("encoder_output", [None]*len(train_dataset))
    train_dataset.add_column("labels", [None]*len(train_dataset))
    train_dataset.add_column("decoder_inputs", [None]*len(train_dataset))
    train_dataset.add_column("encoder_inputs", [None]*len(train_dataset))
    train_dataset = train_dataset.map(mapping_func(encoder, t, tokeniser, text_process=my_func, mcq=mcq_, q_type=q_type))
    train_dataset = train_dataset.remove_columns(["qa_pairs", "filename"])
    #train_dataset = train_dataset.filter(lambda d_: d_["labels"] is not None)
    if save_train:
        train_dataset = train_dataset.filter(lambda d_: d_["split"] == "train")
        train_dataset.save_to_disk(f"/home/egor/Documents/python_codes/visual7w/datasets/v7w/{name}{mcq_}_train")
        for sample in train_dataset:
            print(sample["labels"])
            #print(sample.shape)
            print()
            break
        print(train_dataset.__len__)
    else:
        eval_dataset = train_dataset.filter(lambda d_: d_["split"] == "val")
        eval_dataset.save_to_disk(f"/home/egor/Documents/python_codes/visual7w/datasets/v7w/{name}{mcq_}_val")
        for sample in eval_dataset:
            # print(sample)
            # print()
            break
        print(eval_dataset.__len__)
        eval_dataset.cleanup_cache_files()
    train_dataset.cleanup_cache_files()
    return


def box_map_f(boxes_images, tokeniser):
    def my_f(d_):
        d_["encoder_output"] = boxes_images[d_["box_id"]]
        d_["labels"] = tokeniser(d_["name"], return_tensors="pt").input_ids
        return d_
    return my_f


def boxes_pretraining():
    tokeniser = T5TokenizerFast.from_pretrained("t5-small")
    #tokeniser = PreTrainedTokenizer()
    tokeniser.add_special_tokens(
        {
            "additional_special_tokens": ["[TRUE]", "[FALSE]"]
        }
    )

    boxes_data = Dataset.from_json("/home/egor/Documents/python_codes/visual7w/dataset_v7w_pointing.json", field="boxes")
    images_data = Dataset.from_json("/home/egor/Documents/python_codes/visual7w/dataset_v7w_pointing.json", field="images")

    encoder, t_ = image_encoder(crop_size)
    boxes_images = {}

    print("counting boxes\n")
    boxes_ids = dict()
    deletables = set()
    for image_data in images_data:
        image_id = image_data["image_id"]
        print(f"processing image {image_id}")
        for qa_pair in image_data["qa_pairs"]:
            boxes_ids[qa_pair["answer"]] = image_id
            for choice in qa_pair["multiple_choices"]:
                boxes_ids[choice] = image_id

    print("transforming boxes\n")
    counter = 0
    for box_data in boxes_data:
        box_id = box_data["box_id"]
        image_file = "v7w_" + str(boxes_ids[box_id]) + ".jpg"
        print(f"processing box {box_id}; length {len(boxes_images)}")
        full_image = Image.open(join("/home/egor/Documents/python_codes/visual7w/visual7w_images/images", image_file))
        full_image = full_image.convert("RGB")
        full_w, full_h = full_image.size
        full_image = full_image.resize([image_model_size, image_model_size])
        factor_w = image_model_size / full_w
        factor_h = image_model_size / full_h
        # box_data = boxes_data.filter(lambda x: x["box_id"] == box_id, keep_in_memory=True)[0]
        x_ = int(box_data["x"] * factor_w)
        y_ = int(box_data["y"] * factor_h)
        width_ = int(box_data["width"] * factor_w)
        height_ = int(box_data["height"] * factor_h)
        if (width_ > crop_size * 1.13) or (height_ > crop_size * 1.13):
            continue
        #img_size = max(box_data["width"], box_data["height"])
        # if width_ > height_:
        #     crop_size = max(int(width_ * factor_w), 48)
        # else:
        #     crop_size = max(int(height_ * factor_h), 48)
        cropped_image = full_image.crop([x_, y_, x_ + crop_size, y_ + crop_size])
        try:
            input_ = t_(cropped_image).to("cpu")
            input_ = input_[None, :, :, :]
            # input_ = distortImages(input_)
            out_raw, out_attn = encoder(input_, attn=True)
            boxes_images[box_id] = out_attn
        except RuntimeError:
            print("cancelled")
            deletables.add(box_id)
            continue
            boxes_data.cleanup_cache_files()
        if len(boxes_images)== 50000:
            counter += 1
            print("adding dataset fields")
            boxes_data_ = boxes_data.filter(lambda x: x["box_id"] in boxes_images)
            boxes_data_ = boxes_data_.map(box_map_f(boxes_images, tokeniser))
            print("saving dataset\n")
            boxes_data_.save_to_disk(f"/home/egor/Documents/python_codes/visual7w/datasets/v7w/boxes_{crop_size}_{counter}")
            for sample in boxes_data_:
                for item in sample:
                    print(item)
                    print(sample[item])
                    print()
                    #print(sample.shape)
                print()
                break
            boxes_images = dict()
            images_data.cleanup_cache_files()
            boxes_data.cleanup_cache_files()
            boxes_data_.cleanup_cache_files()
            print()
    boxes_data_ = boxes_data.filter(lambda x: x["box_id"] in boxes_images)
    boxes_data_ = boxes_data_.map(box_map_f(boxes_images, tokeniser))
    boxes_data_.save_to_disk(f"/home/egor/Documents/python_codes/visual7w/datasets/v7w/boxes_{crop_size}_last")
    for sample in boxes_data_:
        for item in sample:
            print(item)
            print(sample[item])
            print()
            #print(sample.shape)
        print()
        break
    images_data.cleanup_cache_files()
    boxes_data.cleanup_cache_files()
    boxes_data_.cleanup_cache_files()
    return

crop_size = 48
image_model_size = 224
data_path = "/home/egor/Documents/python_codes/visual7w/visual7w_images/images"
if __name__ == "__main__":
    # print("pretraining\n")
    # boxes_pretraining()
    # print("\n")
    # print("finetuning\n")
    # do_things(save_train=True, my_func=tokenise_separate_de_en, mcq_=-1, name="where", q_type="where")
    do_things(save_train=True, my_func=tokenise_separate_de_en, mcq_=5, name="where_mcq", q_type="where")
    do_things(save_train=False, my_func=tokenise_separate_de_en, mcq_=5, name="where_mcq", q_type="where")
    # for steps in [0]:
    #     do_things(True, mcq, steps, "mcq")
    # for steps in [0,1,2]:
    #     do_things(True, mcq_new, steps, "mcq_new")
