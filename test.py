import torch
from model import MattingNetwork
from inference import convert_video

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('./pretrained/rvm_mobilenetv3.pth'))

convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='./videos/proj_4.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='./videos/proj_matt_4.mp4', # File path if video; directory path if png sequence.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)