import builtins
original_print = builtins.print
from measurements import get_measurements
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch
from omegaconf import OmegaConf
import clip
import torchaudio
from llama_inference.llama import Tokenizer, Llama
from my_util.misc import NativeScalerWithGradNormCount as NativeScaler
import my_util.misc as misc
new_print = original_print
from codec.MSCodec import MSCodecLM
import random
import typing as tp
from collections import OrderedDict
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from pystoi.stoi import stoi

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

class DenoisingDataset(Dataset):
    def __init__(self, data_root, tsv_path, time, model_path, audio_tokenizer, text_token_embedding, device="cpu", induction=1,vq1_texts=None):
        self.device = device
        self.text_token_embedding = text_token_embedding
        self.data_root = data_root
        self.text_tokenizer = Tokenizer(model_path=model_path + "/tokenizer.model")
        self.induction = induction
        self.vq1_texts = list(vq1_texts)
        self.audio_tokenizer = audio_tokenizer

        self.noisy_paths = []
        self.clean_paths = []
        with open(tsv_path) as f:
            for line in f.readlines():
                image_ids = line.strip('\n').split(",")[:-1]
                curr_clean_paths = []
                curr_noisy_paths = []
                for image_id in image_ids:
                    clean_path, noisy_path = image_id.split("/")
                    curr_clean_paths.append(clean_path)
                    curr_noisy_paths.append(noisy_path)
                self.clean_paths.append(curr_clean_paths)
                self.noisy_paths.append(curr_noisy_paths)
        self.clean_paths = self.clean_paths[time:]
        self.noisy_paths = self.noisy_paths[time:]


    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, index):
        select_texts = []
        select_audios = []
        ###Instruction
        if self.induction == 0:
            instruction = ''
        else:
            #instruction = "Please denoise the last noisy input"
            instruction = (
            "You are an expert in speech enhancement. "
            "Given a noisy separated audio input from a previous speech separation step, "
            "your task is to generate a cleaner and more natural-sounding version of the same speaker's audio. "
            "The input is a quantized representation of audio. "
            "Please output a denoised version using the same quantized format."
            )

        prompt_tokens = torch.tensor(self.text_tokenizer.encode(instruction, bos=True, eos=False)).unsqueeze(0).to(self.device)
        in_tokens = torch.tensor(self.text_tokenizer.encode("###\ninput: < ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        out_tokens = torch.tensor(self.text_tokenizer.encode(" >\noutput: ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        prompt_features = self.text_token_embedding(prompt_tokens)
    
        in_feature = self.text_token_embedding(in_tokens)
        out_feature = self.text_token_embedding(out_tokens)
        noisy_ids = self.noisy_paths[index]
        clean_ids = self.clean_paths[index]
        last_setence = ''
        layer1_len = 0
        layer2_len = 0
        layer3_len = 0
        for i in range(0, len(noisy_ids)):
            noisy_id = noisy_ids[i]
            wav_root = os.path.join(self.data_root, noisy_id)
            wav, sr = torchaudio.load(wav_root)
            if sr != 16000:
                wav = convert_audio(wav, sr, 16000, 1)
            wav = wav.unsqueeze(1).to(self.device)
            if wav.shape[2]/16000 > 1:
                wav = wav[:,:,:1*16000]
            else:
                wav_new = torch.zeros(1, 1, 1*16000).type_as(wav)
                wav_new[:,:,:wav.shape[2]] = wav
                wav = wav_new
            my_code = []
            setence = ''
            with torch.no_grad():
                x, codes , _, _,_,_ = self.audio_tokenizer(wav)
                for kk, code in enumerate(codes):
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[0,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
             
                            real_code = self.text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code += real_code
                            setence += ' ' + str(wo)
                            layer1_len = code.shape[1]
                        else:
                            if kk == 1 : 
                                layer2_len = code.shape[1]   
                             
                            if kk == 2 :
                                layer3_len = code.shape[1]   
                            
                            tmp = code[0,j].item()
                            wo = self.text_tokenizer.decode(tmp)
                            setence += ' ' + str(wo)
                            my_code.append(tmp)
           
            if i == len(noisy_ids)-1:
                last_setence = setence

            my_code = np.array(my_code)
            my_code = torch.from_numpy(my_code).to(self.device)
            select_audios.append(my_code)
            if i != len(noisy_ids)-1:
                select_audios.append(my_code)
            if i == len(noisy_ids)-1:
                print("\nprediction noisy: ", last_setence, "\n") 
                print("\nnoisy id is ", noisy_ids[i], "\n")

        layers_len = [layer1_len, layer2_len, layer3_len]
        for i in range(0, len(clean_ids)):
            clean_id = clean_ids[i]
            wav_root = os.path.join(self.data_root, clean_id)
            wav, sr = torchaudio.load(wav_root)
            if sr != 16000:
                wav = convert_audio(wav, sr, 16000, 1)
            wav = wav.unsqueeze(1).to(self.device)
            if wav.shape[2]/16000 > 1:
                wav = wav[:,:,:1*16000]
            else:
                wav_new = torch.zeros(1, 1, 1*16000).type_as(wav)
                wav_new[:,:,:wav.shape[2]] = wav
                wav = wav_new
            my_code = []
            setence = ''
            with torch.no_grad():
                x, codes , _, _,_,_ = self.audio_tokenizer(wav)
                for kk, code in enumerate(codes):
                    #if kk != 0:
                        #continue
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[0,j].item() # index
                          
                            wo = self.vq1_texts[tmp] # get word
                           
                            real_code = self.text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code += real_code
                            setence += ' ' + str(wo)
                        else:
                            tmp = code[0,j].item()
                            wo = self.text_tokenizer.decode(tmp)
                            setence += ' ' + str(wo)
                            my_code.append(tmp)
                   
            my_code = np.array(my_code)
            my_code = torch.from_numpy(my_code).to(self.device)
            select_texts.append(my_code)
            if i != len(clean_ids)-1:
                select_texts.append(my_code)
     
            if i == len(clean_ids)-1:
                print("\n prediction clean: ", setence, "\n")
                print("\n clean id is ", clean_ids[i], "\n")
            
        ##The last image serves query image (GT)
        target_texts = select_texts[-1] # the last one as the target
        select_texts = select_texts[:-1] # previous

        ##Generating context examples with other images
        for i in range(0, len(select_texts)):
            text_token = select_texts[i].unsqueeze(0)
            text_feature = self.text_token_embedding(text_token)
            vis_token = select_audios[i].unsqueeze(0)
            vis_feature = self.text_token_embedding(vis_token)
            prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens, text_token], dim=-1)
            prompt_features = torch.cat( [prompt_features, in_feature, vis_feature , out_feature, text_feature], dim=1)

        ##Adding query token
        vis_texts = ""
        vis_token = select_audios[-1].unsqueeze(0)
        prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens], dim=-1)
        prompt_features = torch.cat( [prompt_features, in_feature, self.text_token_embedding(vis_token), out_feature], dim=1)
 
        prompt_tokens = prompt_tokens[0].to("cpu")
        prompt_features = prompt_features[0].to("cpu")
        return [prompt_tokens, prompt_features, target_texts, clean_ids, layers_len]

    def old__getitem__(self, index):
        # Use real example from logs
        clean_sound = [15029, 15029, 6336, 18002, 27314, 10298, 29544, 12623, 5858, 9833, 19437, 9563, 1135, 19309, 11408, 24369, 620, 20525, 26429, 19437, 8693, 26429, 263, 8076, 26429, 21649, 4242, 2734, 23483, 403, 1109, 28503, 28883, 6586, 15029, 15029, 3643, 23277, 21431, 3266, 12682, 18411, 27204, 17839, 30576, 8326, 15331, 25982, 23899, 28221, 27528, 3121, 13665, 19184, 9429, 25463, 28198, 3621, 22765, 31970, 20311, 26940, 13993, 22248, 15158, 22005, 31494, 31199, 2220, 13007, 1203, 7610, 3082, 17558, 16872, 30128, 16174, 14759, 31199, 21095, 8319, 22944, 13584, 8778, 9440, 23870, 4779, 1773, 14769, 1938, 14117, 30320, 3497, 2819, 376, 19087, 11309, 27490, 22584, 7379, 29295, 16942, 20103, 16760, 19402, 25327, 31133, 7387, 28062, 27816, 602, 21835, 7568, 22077, 24992, 20017, 15005, 17033, 14515, 15982, 13516, 4825, 14920, 23127, 25098, 19515, 14888, 3808, 25499, 20269, 12360, 28001, 21246, 21966, 25206, 8864, 26098, 26604, 16793, 10438, 1088, 29346, 30002, 26495, 260, 27678, 6688, 29156, 25312, 19979, 27451, 29670, 10711, 16614, 25820, 23548, 2439, 20517, 21946, 13489, 18597, 17926, 29203, 29120, 27842, 16459, 26473, 8148, 19905, 26851, 31175, 24686, 14206, 26846, 1819, 12175, 27622, 18242, 31683, 22067, 19784, 30083, 27233, 8651, 11895, 20711, 23306, 29619, 19455, 26604, 19643, 11444, 30938, 15053, 25923, 6291, 22998, 6072, 11543, 25114, 10089, 2982, 15361, 24041, 13886, 19801, 3236, 17845, 11588, 5376, 21637, 27036, 15980, 579, 25597, 3667, 22097, 31529, 24032, 15265, 30869, 21767, 27938, 14963, 845, 21174, 4948, 20017, 5658, 11935, 30335, 18967, 10256, 9001, 25475, 30875, 9461, 2747, 16419, 22111, 8432]
        noisy_sound = [8830, 8830, 21915, 16082, 13586, 29643, 22301, 28968, 10296, 4203, 2360, 16561, 4783, 21915, 1671, 7417, 13586, 11097, 16766, 4208, 4066, 1623, 15428, 5998, 1554, 1010, 1259, 8329, 14154, 5998, 17770, 8959, 20413, 8830, 8830, 3643, 23277, 21431, 24352, 24618, 30679, 5708, 24841, 23749, 2798, 7744, 29648, 10314, 19066, 21021, 12940, 19327, 29445, 31507, 9686, 24612, 30959, 22497, 7727, 29534, 27238, 594, 10706, 17744, 5092, 10200, 3896, 19004, 28138, 20, 13959, 5438, 16712, 10996, 27101, 16395, 29289, 12721, 3468, 24794, 12948, 9933, 12263, 29632, 29212, 8489, 8016, 31344, 5583, 22018, 18045, 20796, 28610, 23280, 9917, 23871, 28745, 20917, 24442, 29295, 10154, 20103, 16760, 19402, 7218, 30465, 30145, 8499, 9318, 21872, 2920, 19022, 7635, 9015, 6053, 12887, 23108, 8604, 21187, 16947, 11222, 28687, 19780, 23822, 28358, 24390, 8132, 7429, 21074, 16494, 13480, 20023, 5961, 608, 15067, 13214, 20336, 2368, 15657, 8795, 2775, 14198, 17417, 1418, 8746, 23198, 17625, 9766, 24350, 18617, 10424, 31680, 1389, 22512, 28023, 972, 30415, 18600, 25044, 31631, 29068, 11825, 16946, 2000, 30923, 16561, 29292, 26643, 16172, 18491, 21815, 3274, 22697, 3879, 15617, 3969, 16290, 21252, 6694, 2508, 2721, 15765, 19746, 5393, 4181, 1280, 30913, 7292, 6113, 12155, 23427, 3215, 5107, 26447, 2983, 8305, 23548, 10407, 16371, 30747, 18200, 10052, 31732, 3809, 25221, 28103, 24578, 24097, 22134, 27814, 26221, 2837, 31990, 8035, 12157, 21005, 23783, 23723, 23068, 18561, 13716, 14301, 30501, 30870, 26543, 26962, 20431, 6382, 6231, 24728, 29280, 10256, 10428, 2484, 23049, 23049, 2747, 16419, 22111, 8432]
        instruction = f"Convert noisy audio to clean:\nExample: Noisy {str(noisy_sound)} -> Clean {str(clean_sound)}\nNoisy codes: "
        prompt_tokens = torch.tensor(self.text_tokenizer.encode(instruction, bos=True, eos=False)).unsqueeze(0).to(self.device)
        prompt_features = self.text_token_embedding(prompt_tokens)

        noisy_path = os.path.join(self.data_root, self.noisy_paths[index])
        clean_path = os.path.join(self.data_root, self.clean_paths[index])
        noisy_wav, sr = torchaudio.load(noisy_path)
        clean_wav, sr_clean = torchaudio.load(clean_path)

        if sr != 16000:
            noisy_wav = convert_audio(noisy_wav, sr, 16000, 1)
        if sr_clean != 16000:
            clean_wav = convert_audio(clean_wav, sr_clean, 16000, 1)

        for wav, name in [(noisy_wav, "noisy"), (clean_wav, "clean")]:
            if len(wav.shape) < 2 or wav.shape[1] == 0:
                raise ValueError(f"Invalid {name} audio shape: {wav.shape}")
            if wav.shape[1] < 16000:
                wav_new = torch.zeros(1, 16000).type_as(wav)
                wav_new[:, :wav.shape[1]] = wav
                wav = wav_new
            if name == "noisy":
                noisy_wav = wav
            else:
                clean_wav = wav

        noisy_wav = noisy_wav.unsqueeze(1).to(self.device)
        clean_wav = clean_wav.unsqueeze(1).to(self.device)

        with torch.no_grad():
            _, noisy_codes, _, _, _, _ = self.audio_tokenizer(noisy_wav)
            _, clean_codes, _, _, _, _ = self.audio_tokenizer(clean_wav)
            noisy_codes_flat = noisy_codes[0][0]
            clean_codes_flat = [codes[0] for codes in clean_codes]

        noisy_feature = self.text_token_embedding(noisy_codes_flat.unsqueeze(0))
        prompt_tokens = torch.cat([prompt_tokens, noisy_codes_flat.unsqueeze(0)], dim=-1)
        prompt_features = torch.cat([prompt_features, noisy_feature], dim=1)

        prompt_tokens = prompt_tokens[0].to("cpu")
        prompt_features = prompt_features[0].to("cpu")
        target_codes = [code.to("cpu") for code in clean_codes_flat]
        return [prompt_tokens, prompt_features, target_codes, noisy_path, clean_path]

def custom_collate_fn(batch):
    prompt_tokens = torch.stack([item[0] for item in batch])
    prompt_features = torch.stack([item[1] for item in batch])
    target_codes = [torch.stack([item[2][i] for item in batch]) for i in range(3)]
    noisy_paths = [item[3] for item in batch]
    clean_paths = [item[4] for item in batch]
    return [prompt_tokens, prompt_features, target_codes, noisy_paths, clean_paths]

def get_args_parser():
    parser = argparse.ArgumentParser("Audio Denoising", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--llama_model_path", default="./llama", type=str)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--audio_path", default="/data/all", type=str)
    parser.add_argument("--file_path", default="/data/all", type=str)
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", type=str)
    parser.add_argument("--n_vision_words", default=32000, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str)
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml")
    parser.add_argument("--codec_ckpt", type=str, default="stage_1_llama_fix-40.pth")
    parser.add_argument("--induction", type=int, default=2)
    return parser

    
def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True
    vq1_texts = np.load("./layer1.npy", allow_pickle=True)

    exp_model_config = OmegaConf.load(args.vq_config_path)
    model = MSCodecLM(**exp_model_config.generator.config)
    parameter_dict = torch.load(args.codec_ckpt, weights_only=False)
    model.load_state_dict(parameter_dict['codec_model'])
    model.to(device)
    model.eval()

    generator = Llama.build(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.llama_model_path + "/tokenizer.model",
        max_seq_len=args.max_seq_len,
        max_batch_size=2,
    )
    llama_vocab_size = generator.tokenizer.n_words
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    #llama_vocab_size = generator.tokenizer.n_words
    #print(f"LLaMA vocabulary size: {llama_vocab_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = DenoisingDataset(
        data_root=args.audio_path, tsv_path=args.file_path, time=0,
        model_path=args.llama_model_path, audio_tokenizer=model,
        text_token_embedding=generator.model.tok_embeddings, device=device,
        induction=args.induction, vq1_texts=vq1_texts
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=False
    )

    total_si_sdr = 0
    total_samples = 0
    total_stoi = 0
    total_pesq = 0
    metric_logger = misc.MetricLogger(delimiter="  ")
    i = 0
    max_sdr_value = float('-inf')
    max_sdr_reconstructed = ""
    max_sdr_clean = ""
    max_snr_value = float('-inf')
    max_snr_reconstructed = ""
    max_snr_clean = ""
    max_stoi_value = float('-inf')
    max_stoi_reconstructed = ""
    max_stoi_clean = ""
    """for data_iter_step, [prompt_tokens, prompt_features, target_texts, clean_ids, layers_len] in enumerate(
        metric_logger.log_every(data_loader, 10, "Denoising")
    ):"""
    for data_iter_step, [prompt_tokens, prompt_features, target_texts, clean_ids, layers_len] in enumerate(
        metric_logger.log_every(data_loader, 10, "Denoising")
    ):
        
        prompt_tokens = prompt_tokens.to(device)
        prompt_features = prompt_features.to(device)

       
        out_mask = torch.ones(args.batch_size, args.max_seq_len, llama_vocab_size).to(device)
        predictions = generator.generate_fewshot(
            prompt_tokens,
            prompt_features,
            induction=args.induction,
            out_mask=out_mask,
            max_gen_len=100,
            temperature=0,
            top_p=1.0,
        )
        clean_p = clean_ids[-1][0]
        for idx, (pred, target_text) in enumerate(zip(predictions, target_texts)):
            pred_tokens = pred['tokens']  # Extract predicted tokens
            # Step 2: Decode LLaMA tokens using LLaMA's text tokenizer to get words
            decoded_words = [generator.tokenizer.decode(t) for t in pred_tokens]
            #print("length of decoded ", len(decoded_words))
            stop_token = "###"
            # Find the index of stop token
            if stop_token in decoded_words:
                stop_index = decoded_words.index(stop_token)
                decoded_words = decoded_words[:stop_index]
            # Step 3: Map each word back to its corresponding VQ token
            layers = [t.item() for t in layers_len]
            try : 
                vq1_indices = [dataset.vq1_texts.index(word) for word in decoded_words[:layers[0]]]  # Map words to VQ indices
            except ValueError:
                print("couldnt find word in vq1_texts. continuing")
                continue
            vq1 = torch.tensor(vq1_indices).unsqueeze(0).to(device)
            vq2 = torch.tensor(pred_tokens[layers[0]:layers[0]+layers[1]]).unsqueeze(0).to(device)
            vq3 = torch.tensor(pred_tokens[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]]).unsqueeze(0).to(device)
            # Step 4: Convert the list of VQ indices into a tensor and move it to the device
              # Add batch dimension
            # Step 5: Decode the VQ tokens back into the waveform using the audio tokenizer
            tensor_list = [vq1, vq2, vq3]
            with torch.no_grad():
                reconstructed_wav, *_ = model.decode(tensor_list)
                # Save the output wav file
                decoded_audio = reconstructed_wav.squeeze(1)
                number_part = ""
                if clean_p.startswith("s1"):
                    number_part = clean_p.split('s1-')[1]
                if clean_p.startswith("s2"):
                    number_part = clean_p.split('s2-')[1]
                file_name = f'recontructed-{number_part}.wav'
                output_path = os.path.join(args.output_dir, file_name)
                try:
                    torchaudio.save(output_path, decoded_audio.cpu(), sample_rate=16000)
                    print(f"Saved Separated audio to: {output_path}")
                except Exception as e:
                    print(f"Failed to save Separated audio: {e}")
                i+=1
            # Compute SI-SDR
            clean_wav_root = os.path.join(dataset.data_root, clean_p)
            clean_wav, clean_sr = torchaudio.load(clean_wav_root)
            if clean_sr != 16000:
                clean_wav = convert_audio(clean_wav, clean_sr, 16000, 1)
            clean_length = clean_wav.shape[1]
            clean_wav = clean_wav.unsqueeze(1).to(device)
            min_length = min(decoded_audio.shape[1], clean_wav.shape[1])
            decoded_audio = decoded_audio[:, :min_length]
            clean_wav_eval = clean_wav.squeeze(1)[:, :min_length]
            builtins.print = original_print
            sdr ,p, s = get_measurements(clean_wav_root, output_path)
            builtins.print = new_print
            
            
            clean_wav_np = clean_wav_eval.cpu().numpy()  # Move to CPU and convert to NumPy
            decoded_audio_np = decoded_audio.cpu().numpy()  # Move to CPU and convert to NumPy
            total_si_sdr += sdr
            total_stoi += s
            total_pesq = p
            total_samples += 1
            

            print(f"Clean_p: {clean_p}, SI-SDR: {sdr:.4f} dB, PESQ: {p:.2f}, STOI: {s:.2f}Saved to: {output_path} ")

    avg_si_sdr = total_si_sdr / total_samples if total_samples > 0 else 0
    avg_pesq = total_pesq / total_samples if total_samples > 0 else 0
    avg_stoi = total_stoi / total_samples if total_samples > 0 else 0
    print(f"Average SI-SDR: {avg_si_sdr:.4f} dB\n")
    print(f"Average PESQ: {avg_pesq:.4f} dB\n")
    print(f"Average stoi: {avg_stoi:.4f} dB\n")
    print(f"MAX SDR value: {max_sdr_value:.4f} dB, for the reconstructed file: {max_sdr_reconstructed}\n and clean: {max_sdr_clean}\n")
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.batch_size = 1
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)