import random

import tqdm
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

from utils.salsa_utils.salsa_dataloader import SALSA_CAPTIONS
import pickle
os.chdir('Motion-Agent-Salsa') # to refine the data loader felan.

from utils.salsa_utils.libs.MotionScript.ms_utils_visu import render_HQ_Salsa
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval4Ahmet"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    model.load_model('ckpt/motionllm.pth')

    model.llm.eval()
    model.llm.cuda()



    for style in SALSA_CAPTIONS:
        # if style != 'professional': continue
        for iterate in tqdm.tqdm(range(1)):

            motion_tokens_to_generate = []
            captions_list = []
            for t in range(3):
                caption = random.choice(SALSA_CAPTIONS[style])
                captions_list.append(caption)
                motion_tokens = model.generate(caption)
                motion_tokens_to_generate.append(motion_tokens)

            motion_tokens = torch.cat(motion_tokens_to_generate)
            motion = model.net.forward_decoder(motion_tokens)
            motion = model.denormalize(motion.detach().cpu().numpy())

            positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
            # print(motion.shape)


            if iterate < 5:
                sav_path = f"./demo/eval4Ahmet/{style}"

                os.makedirs(sav_path, exist_ok=True)
                plot_3d_motion(os.path.join(sav_path,
                                f"motionllm_{style}_{iterate}.mp4"),
                                t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                                title='\n'.join(captions_list),
                                fps=20, radius=4)


                vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                                sav_path,
                                name=f'motionllm_{style}_{iterate}_3DMesh.mp4')

                # For Shay
                my_dict = {'caption': caption,
                           'HML3D_vec': motion.squeeze(),
                           'positions': positions.squeeze().detach().cpu().numpy(),
                           'vertuces': vertices,
                           'faces': faces}
                with open(f"{sav_path}/motionllm_{style}_{iterate}.pk", "wb") as f:
                    pickle.dump(my_dict, f)

if __name__ == "__main__":

    # motion_agent_demo()

    motionllm_evaluation_qualitative()