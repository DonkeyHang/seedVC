import os
import sys
import time
import argparse
import warnings
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
import librosa
from tqdm import tqdm
from modules.commons import *
import torchaudio.compliance.kaldi as kaldi
from hf_utils import load_custom_model_from_hf
from modules.commons import str2bool

warnings.simplefilter("ignore")

# 设置设备
device = None

# 全局变量
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3  # 默认参考音频长度，单位为秒
ce_dit_difference = 2.0  # 默认内容编码器与DiT的时间差，单位为秒
fp16 = False

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0):
    """
    流式推理函数
    """
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    
    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"设置ce_dit_difference为{cd_difference}秒")
    
    # 如果参考音频变化或首次运行，处理参考音频
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"设置最大参考长度为{max_prompt_length}秒")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        # 处理参考音频
        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    # 时间测量
    if device.type == "mps":
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        torch.mps.synchronize()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

    # 语义特征提取
    start_event.record()
    S_alt = semantic_fn(input_wav_res.unsqueeze(0))
    end_event.record()
    
    if device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if prompt_condition is None:
        print(f"语义特征提取耗时: {elapsed_time_ms}ms")

    # 应用内容编码器与DiT的时间差
    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    
    # 生成条件
    cond = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    # 使用条件流匹配进行推理
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target).squeeze()
    
    # 裁剪输出到所需长度
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

def load_models(args):
    """
    加载模型
    """
    global fp16
    fp16 = args.fp16
    
    # 设置默认检查点和配置文件路径
    model_filename = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
    config_filename = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    
    # 尝试多种路径寻找模型文件
    if args.checkpoint is None:
        # 首先检查本地checkpoints目录
        local_checkpoint = os.path.join("./checkpoints", model_filename)
        local_config = os.path.join("./checkpoints", config_filename)
        
        # 如果文件存在就使用本地文件
        if os.path.exists(local_checkpoint) and os.path.exists(local_config):
            print(f"✓ 使用本地模型文件: {local_checkpoint}")
            dit_checkpoint_path = local_checkpoint
            dit_config_path = local_config
        else:
            # 尝试从HF下载
            print(f"本地文件不存在，尝试从Hugging Face下载: {model_filename}")
            try:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    model_filename,
                    config_filename
                )
                print(f"✓ 成功下载模型文件")
            except Exception as e:
                print(f"✗ 下载模型文件失败: {e}")
                print("\n请尝试以下方法之一:")
                print("1. 运行 'python download_model.py' 下载所有必要文件")
                print("2. 手动下载文件并放入 './checkpoints' 目录:")
                print(f"   - 模型文件: https://huggingface.co/Plachta/Seed-VC/resolve/main/{model_filename}")
                print(f"   - 配置文件: https://huggingface.co/Plachta/Seed-VC/resolve/main/{config_filename}")
                print("3. 使用 --checkpoint 和 --config 参数指定本地文件路径")
                raise RuntimeError("无法加载模型文件，请参考上述建议")
    else:
        # 使用命令行参数指定的文件
        dit_checkpoint_path = args.checkpoint
        dit_config_path = args.config
        print(f"✓ 使用指定的模型文件: {dit_checkpoint_path}")
    
    # 加载配置
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # 加载检查点
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # 加载CAMPPlus声纹提取器
    from modules.campplus.DTDNN import CAMPPlus
    
    # 首先检查本地路径
    local_campplus = "./checkpoints/campplus_cn_common.bin"
    if os.path.exists(local_campplus):
        print(f"✓ 使用本地声纹提取器: {local_campplus}")
        campplus_ckpt_path = local_campplus
    else:
        # 尝试从HF下载
        print("本地声纹提取器不存在，尝试从Hugging Face下载")
        try:
            campplus_ckpt_path = load_custom_model_from_hf(
                "funasr/campplus", "campplus_cn_common.bin", config_filename=None
            )
            print("✓ 成功下载声纹提取器")
        except Exception as e:
            print(f"✗ 下载声纹提取器失败: {e}")
            print("\n请手动下载声纹提取器:")
            print("- 下载链接: https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin")
            print("- 保存到: ./checkpoints/campplus_cn_common.bin")
            raise RuntimeError("无法加载声纹提取器")
    
    print("正在加载声纹提取器...")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # 加载声码器
    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # 移除模型中的weight norm并设置为eval模式
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        
        # 首先检查本地路径
        local_hift = "./checkpoints/hift.pt"
        if os.path.exists(local_hift):
            print(f"✓ 使用本地hift.pt文件: {local_hift}")
            hift_path = local_hift
        else:
            # 尝试从HF下载
            print("本地hift.pt文件不存在，尝试从Hugging Face下载")
            try:
                hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
                print("✓ 成功下载hift.pt")
            except Exception as e:
                print(f"✗ 下载hift.pt失败: {e}")
                print("\n请手动下载hift.pt:")
                print("- 下载链接: https://huggingface.co/FunAudioLLM/CosyVoice-300M/resolve/main/hift.pt")
                print("- 保存到: ./checkpoints/hift.pt")
                raise RuntimeError("无法加载hift.pt文件")
                
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"声码器模型总参数量: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"未知声码器类型: {vocoder_type}")

    # 加载语音内容编码器
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        
        # 检查本地路径
        model_id = whisper_name.split('/')[-1] if '/' in whisper_name else whisper_name
        local_whisper_dir = f"./checkpoints/{model_id}"
        os.makedirs(local_whisper_dir, exist_ok=True)
        
        print(f"正在加载Whisper模型: {whisper_name}")
        
        try:
            # 尝试从本地加载
            try:
                print(f"尝试从本地加载Whisper模型: {local_whisper_dir}")
                whisper_model = WhisperModel.from_pretrained(
                    local_whisper_dir, 
                    torch_dtype=torch.float16,
                    local_files_only=True
                ).to(device)
                print("✓ 成功从本地加载Whisper模型")
                
                whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
                    local_whisper_dir,
                    local_files_only=True
                )
                print("✓ 成功从本地加载Whisper特征提取器")
            except Exception as e:
                print(f"✗ 从本地加载Whisper模型失败: {e}")
                print(f"正在尝试从Hugging Face下载...")
                
                try:
                    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
                    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
                    
                    # 保存到本地以便下次使用
                    print(f"保存Whisper模型到本地: {local_whisper_dir}")
                    whisper_model.save_pretrained(local_whisper_dir)
                    whisper_feature_extractor.save_pretrained(local_whisper_dir)
                except Exception as e:
                    print(f"✗ 下载Whisper模型失败: {e}")
                    print(f"\n请手动下载Whisper模型:")
                    print(f"1. 访问: https://huggingface.co/{whisper_name}")
                    print(f"2. 下载模型文件并放入 {local_whisper_dir} 目录")
                    raise RuntimeError("无法加载Whisper模型")
            
            del whisper_model.decoder
            
            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                       return_tensors="pt",
                                                       return_attention_mask=True)
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori
                
        except Exception as e:
            print(f"加载Whisper模型失败: {e}")
            raise RuntimeError("无法加载Whisper模型")
            
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        
        # 获取模型ID，用于本地文件命名
        model_id = hubert_model_name.split('/')[-1] if '/' in hubert_model_name else hubert_model_name
        local_hubert_dir = f"./checkpoints/{model_id}"
        os.makedirs(local_hubert_dir, exist_ok=True)
        
        print(f"正在加载HuBERT模型: {hubert_model_name}")
        
        try:
            # 尝试从本地加载
            try:
                print(f"尝试从本地加载HuBERT模型: {local_hubert_dir}")
                hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    local_hubert_dir, 
                    local_files_only=True
                )
                hubert_model = HubertModel.from_pretrained(
                    local_hubert_dir,
                    local_files_only=True
                )
                print("✓ 成功从本地加载HuBERT模型")
            except Exception as e:
                print(f"✗ 从本地加载HuBERT模型失败: {e}")
                print(f"\n请手动下载HuBERT模型:")
                print(f"1. 访问: https://huggingface.co/{hubert_model_name}")
                print(f"2. 下载模型文件并放入 {local_hubert_dir} 目录")
                raise RuntimeError("无法加载HuBERT模型")
            
            hubert_model = hubert_model.to(device)
            hubert_model = hubert_model.eval()
            hubert_model = hubert_model.half()
            
            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy()
                    for bib in range(len(waves_16k))
                ]
                ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                    return_tensors="pt",
                                                    return_attention_mask=True,
                                                    padding=True,
                                                    sampling_rate=16000).to(device)
                with torch.no_grad():
                    ori_outputs = hubert_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
                
        except Exception as e:
            print(f"加载HuBERT模型失败: {e}")
            raise RuntimeError("无法加载HuBERT模型")
            
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        
        # 获取模型ID，用于本地文件命名
        model_id = model_name.split('/')[-1] if '/' in model_name else model_name
        local_model_dir = f"./checkpoints/{model_id}"
        os.makedirs(local_model_dir, exist_ok=True)
        
        print(f"正在加载XLSR模型: {model_name}")
        
        try:
            # 尝试从本地加载
            try:
                print(f"尝试从本地加载XLSR模型: {local_model_dir}")
                wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    local_model_dir, 
                    local_files_only=True
                )
                wav2vec_model = Wav2Vec2Model.from_pretrained(
                    local_model_dir,
                    local_files_only=True
                )
                print("✓ 成功从本地加载XLSR模型")
            except Exception as e:
                print(f"✗ 从本地加载XLSR模型失败: {e}")
                print(f"\n请手动下载XLSR模型:")
                print(f"1. 访问: https://huggingface.co/{model_name}")
                print(f"2. 下载模型文件并放入 {local_model_dir} 目录")
                raise RuntimeError("无法加载XLSR模型")
            
            # 限制encoder层数
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model = wav2vec_model.to(device)
            wav2vec_model = wav2vec_model.eval()
            wav2vec_model = wav2vec_model.half()
            
            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy()
                    for bib in range(len(waves_16k))
                ]
                ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                    return_tensors="pt",
                                                    return_attention_mask=True,
                                                    padding=True,
                                                    sampling_rate=16000).to(device)
                with torch.no_grad():
                    ori_outputs = wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
                
        except Exception as e:
            print(f"加载XLSR模型失败: {e}")
            print("\n请尝试以下步骤手动下载模型文件:")
            print(f"1. 创建目录: mkdir -p {local_model_dir}")
            print(f"2. 下载所有模型文件:")
            print(f"   • 访问: https://huggingface.co/{model_name}")
            print(f"   • 下载所有模型文件并放入 {local_model_dir} 目录")
            print(f"3. 然后重新运行程序")
            raise RuntimeError("无法加载XLSR模型")
    else:
        raise ValueError(f"未知语音内容编码器类型: {speech_tokenizer_type}")
        
    # 生成mel频谱图
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

def crossfade(chunk1, chunk2, overlap):
    """
    将两个音频块进行交叉淡化
    """
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

@torch.no_grad()
def main(args):
    """
    主函数 - 使用流式处理方式处理音频文件
    """
    global device
    start_time = time.time()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu else "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model_set = load_models(args)
    sr = model_set[-1]["sampling_rate"]
    
    # 加载源音频和参考音频
    print(f"加载源音频: {args.source}")
    source_audio, _ = librosa.load(args.source, sr=sr)
    print(f"加载参考音频: {args.target}")
    reference_audio, _ = librosa.load(args.target, sr=sr)
    
    # 如果参考音频超过最大长度，截断
    max_ref_length = int(args.max_prompt_length * sr)
    if len(reference_audio) > max_ref_length:
        reference_audio = reference_audio[:max_ref_length]
    
    # 打印音频信息
    print(f"源音频长度: {len(source_audio)/sr:.2f}秒, 采样率: {sr}Hz")
    print(f"参考音频长度: {len(reference_audio)/sr:.2f}秒, 采样率: {sr}Hz")
    
    # 设置块大小和交叉淡化参数
    zc = sr // 50  # 基本时间单位
    block_frame = int(np.round(args.block_time * sr / zc)) * zc
    block_frame_16k = 320 * block_frame // zc
    crossfade_frame = int(np.round(args.crossfade_time * sr / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(args.extra_time_ce * sr / zc)) * zc
    extra_frame_right = int(np.round(args.extra_time_right * sr / zc)) * zc
    
    # 确保源音频长度足够
    if len(source_audio) < block_frame + extra_frame + extra_frame_right:
        pad_length = block_frame + extra_frame + extra_frame_right - len(source_audio)
        source_audio = np.pad(source_audio, (0, pad_length), 'constant')
    
    # 初始化SOLA算法的缓冲区
    sola_buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)
    
    # 计算处理参数
    skip_head = extra_frame // zc
    skip_tail = extra_frame_right // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    
    # 创建淡入淡出窗口
    fade_in_window = (
        torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device, dtype=torch.float32)) ** 2
    )
    fade_out_window = 1 - fade_in_window
    
    # 准备输出数组
    output_chunks = []
    total_blocks = 0
    total_infer_time = 0
    
    # 分块处理音频
    input_buffer = np.zeros(extra_frame + crossfade_frame + sola_search_frame + block_frame + extra_frame_right, dtype=np.float32)
    
    # 计算总块数
    total_frames = len(source_audio)
    num_blocks = (total_frames + block_frame - 1) // block_frame
    
    # 使用tqdm显示进度
    print(f"开始流式处理，块大小: {block_frame/sr:.3f}秒, 总块数: {num_blocks}")
    pbar = tqdm(total=num_blocks, desc="处理进度")
    
    for i in range(0, total_frames, block_frame):
        # 准备当前块
        end_idx = min(i + block_frame, total_frames)
        current_block = source_audio[i:end_idx]
        
        # 如果当前块不足一个块大小，进行填充
        if len(current_block) < block_frame:
            current_block = np.pad(current_block, (0, block_frame - len(current_block)), 'constant')
        
        # 更新输入缓冲区
        input_buffer[:-block_frame] = input_buffer[block_frame:]
        input_buffer[-block_frame:] = current_block
        
        # 转换为torch张量并移至设备
        input_wav = torch.from_numpy(input_buffer).to(device, dtype=torch.float32)
        
        # 重采样到16kHz
        input_wav_res = torch.from_numpy(
            librosa.resample(input_buffer, orig_sr=sr, target_sr=16000)
        ).to(device, dtype=torch.float32)
        
        # 执行语音转换
        infer_start = time.time()
        infer_wav = custom_infer(
            model_set,
            reference_audio,
            args.target,
            input_wav_res,
            block_frame_16k,
            skip_head,
            skip_tail,
            return_length,
            args.diffusion_steps,
            args.inference_cfg_rate,
            args.max_prompt_length,
            args.extra_time_ce - args.extra_time,
        )
        infer_time = time.time() - infer_start
        total_infer_time += infer_time
        total_blocks += 1
        
        # SOLA算法处理
        conv_input = infer_wav[None, None, :sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, sola_buffer_frame, device=device),
            ) + 1e-8
        )
        
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:
            if device.type == "mps":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = 0
            
        # 应用交叉淡化
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:sola_buffer_frame] *= fade_in_window
        infer_wav[:sola_buffer_frame] += sola_buffer * fade_out_window
        
        # 更新SOLA缓冲区
        sola_buffer[:] = infer_wav[block_frame:block_frame + sola_buffer_frame]
        
        # 添加处理后的块到输出
        output_chunks.append(infer_wav[:block_frame].cpu().numpy())
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 合并所有输出块
    final_output = np.concatenate(output_chunks)
    
    # 裁剪输出到与源音频相同的长度
    final_output = final_output[:total_frames]
    
    # 保存结果
    output_path = os.path.join(args.output, f"realtime_vc_{os.path.basename(args.source)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, torch.tensor(final_output).unsqueeze(0), sr)
    
    # 打印处理统计信息
    total_time = time.time() - start_time
    print(f"处理完成，总用时: {total_time:.2f}秒")
    print(f"平均每块推理时间: {total_infer_time / total_blocks * 1000:.1f}毫秒")
    print(f"实时系数 (RTF): {total_infer_time / (len(source_audio) / sr):.3f}")
    print(f"输出文件保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用流式处理进行语音转换")
    
    # 基本参数
    parser.add_argument("--source", type=str, default="./examples/source/jay_0.wav", help="源音频文件路径")
    parser.add_argument("--target", type=str, default="./examples/reference/azuma_0.wav", help="参考音频文件路径")
    parser.add_argument("--output", type=str, default="./reconstructed", help="输出目录")
    
    # 模型参数
    # parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    # parser.add_argument("--config", type=str, default=None, help="模型配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="", default="./checkpoints/DiT_uvit_tat_xlsr_ema.pth")#origin
    parser.add_argument("--config", type=str, help="", default="./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml")#origin
    parser.add_argument("--diffusion-steps", type=int, default=10, help="扩散步数")
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7, help="推理CFG率")
    
    # 流式处理参数
    # parser.add_argument("--block-time", type=float, default=0.18, help="块时间(秒)")
    parser.add_argument("--block-time", type=float, default=0.30, help="块时间(秒)")
    # parser.add_argument("--crossfade-time", type=float, default=0.04, help="交叉淡化时间(秒)")
    parser.add_argument("--crossfade-time", type=float, default=0.10, help="交叉淡化时间(秒)")
    # parser.add_argument("--max-prompt-length", type=float, default=3.0, help="最大参考音频长度(秒)")
    parser.add_argument("--max-prompt-length", type=float, default=5.0, help="最大参考音频长度(秒)")
    parser.add_argument("--extra-time", type=float, default=0.5, help="额外DiT上下文时间(秒)")#0.5 feeling good
    parser.add_argument("--extra-time-ce", type=float, default=2.5, help="额外内容编码器上下文时间(秒)")# is not have too much diff,i suggessed for 1.0
    parser.add_argument("--extra-time-right", type=float, default=0.02, help="右侧额外上下文时间(秒)")
    
    # 其他参数
    parser.add_argument("--fp16", type=str2bool, default=True, help="是否使用fp16")
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU ID")
    
    args = parser.parse_args()
    main(args)