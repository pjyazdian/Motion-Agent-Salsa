from openai import AzureOpenAI
from models.motion_agent import MotionAgent
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import torch

import os
import sys

# Path to the FFmpeg bin directory
ffmpeg_path = r'S:\Payam\LAMMA\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin'
import matplotlib as mpl
import shutil

ffmpeg_path = shutil.which("ffmpeg")
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path


# Check if the path is already in the PATH variable
if ffmpeg_path not in os.environ['PATH']:
    # Add the path to the PATH environment variable
    os.environ['PATH'] += os.pathsep + ffmpeg_path



def motion_agent_demo():
    # Initialize the client
    client = AzureOpenAI(
        api_key="DDJ7XQo5NWYSxRGwOecZNt7VOwaEEBKWh8eJVWS7YqyCGarpMlviJQQJ99BDACBsN54XJ3w3AAABACOGm6nK", # your api key
        api_version="2024-10-21",
        azure_endpoint="https://salsa.openai.azure.com/" # your azure endpoint
    )

    endpoint = "https://pjome-m9a36rct-eastus2.cognitiveservices.azure.com/"
    model_name = "gpt-35-turbo"
    deployment = "gpt-35-turbo-Salsa"
    subscription_key = "uPN20WiZ0suB5YVbTkwkVjT8Rqdwudlg0fqMaZGu9uF9pQY5oHI4JQQJ99BDACHYHv6XJ3w3AAAAACOGtmqx"
    api_version = "2024-12-01-preview"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # No Azure:
    from openai import OpenAI
    API_KEY = "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"
    client = OpenAI(api_key=API_KEY)

    args = get_args_parser()
    args.save_dir = "./demo"
    args.device = 'cuda:0'

    motion_agent = MotionAgent(args, client)
    motion_agent.chat()

def motionllm_demo():
    model = MotionLLM(get_args_parser())
    model.load_model('ckpt/motionllm.pth')
    model.llm.eval()
    model.llm.cuda()
    
    caption = 'A man is doing cartwheels.'
    motion = model.generate(caption)

    motion = model.denormalize(motion.detach().cpu().numpy())
    motion = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
    print(motion.shape)
    plot_3d_motion(f"motionllm_demo.mp4", t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=caption, fps=20, radius=4)

if __name__ == "__main__":
    motion_agent_demo()
