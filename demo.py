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
from models.training_utils import *
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

from utils.salsa_utils.libs.MotionScript.ms_utils_visu import render_HQ_Salsa, render_HQ_Salsa_pair
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

                    '''
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval_new"
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
                    '''


def motionllm_evaluation_qualitative_pairs():


    args = get_args_parser()
    args.save_dir = "./demo/eval4Ahmet"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    # model.load_model('output_trained/follower_to_leader_v2/Xmotionllm_epoch13.pth')??
    model.load_model('output_trained\pretrain_all/Xmotionllm_epoch5.pth')
    model.load_model('output_trained/follower_to_leader_v3/Xmotionllm_epoch10.pth') # 'follower_to_leader'

    model.llm.eval()
    model.llm.cuda()

    motion_tokens_to_generate = []
    the_other_motion_tokens = []

    Style= 'beginner' # 'professional'
    Pair="Pair1"
    s, e = 0, 10
    items = load_data_pair(style=Style, pair=Pair, start_sec=s, end_sec=e)


    for item in items:

        level, ms_desc_L, ms_des_F, \
            vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = item
        current_batch_task = 'follower_to_leader'


        # captions_list = []


        full_prompt, input_ids = build_random_training_instance_salsa_prompt(
            tokenizer=model.tokenizer,
            leader_motion_script_segments=ms_desc_L.split('-->'),
            follower_motion_script_segments=ms_des_F.split('-->'),
            leader_motion_tokens=vq_tokens_L,
            follower_motion_tokens=vq_tokens_F,
            audio_tokens=audio_tokens,
            proficiency_level=level,
            allowed_tasks=[current_batch_task],
            snippet_prob=0.5,
            min_snippet_steps=1,
            max_snippet_steps=4,
            inference=True
        )



        motion_tokens = model.generate_Payam(full_prompt, input_ids)
        # motion_tokens = vq_tokens_L
        motion_tokens_to_generate.append(motion_tokens)

        the_other_motion_tokens.append(vq_tokens_F if current_batch_task=='follower_to_leader' else vq_tokens_F)
    print("Inference completed.\nExporting results...")
    #Todo ------------High priority---------------------
    # We need to feed the first frame to keep continuity and etc.
    motion_tokens = torch.cat(motion_tokens_to_generate)
    motion = model.net.forward_decoder(motion_tokens)
    motion = model.denormalize(motion.detach().cpu().numpy())
    positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
    # print(motion.shape)

    # Leader:
    the_other_motion_tokens = torch.cat(the_other_motion_tokens)
    the_other_motion = model.net.forward_decoder(the_other_motion_tokens)
    the_other_motion = model.denormalize(the_other_motion.detach().cpu().numpy())
    the_other_positions = recover_from_ric(torch.from_numpy(the_other_motion).float().cuda(), 22)
    # print(motion.shape)

    sav_path = f"./demo/eval4pair/{level}"
    iterate = 0
    # Todo: test two people mesh:
    # just for follower to leader task for now
    leader_positions, follower_positions = the_other_positions, positions

    vertices1, faces1, vertices2, faces2 = render_HQ_Salsa_pair(leader_positions.squeeze().detach().cpu().numpy(),
                                                                follower_positions.squeeze().detach().cpu().numpy(),
                                                                  sav_path,
                                                                  name=f'motionllm_{level}_{iterate}_3DMesh_predicted.mp4')


    iterate = 0
    if True:
        sav_path = f"./demo/eval4pair/{level}"

        os.makedirs(sav_path, exist_ok=True)
        plot_3d_motion(os.path.join(sav_path,
                        f"motionllm_{level}_{iterate}_predicted.mp4"),
                        t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                        title=(f"Generated {'leader' if current_batch_task=='follower_to_leader' else 'follower'}: {Style} {Pair}"),
                        fps=20, radius=4)


        vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                        sav_path,
                        name=f'motionllm_{level}_{iterate}_3DMesh_predicted.mp4')
        vertices, faces = None, None
        # For Shay
        my_dict = {'caption': "infer",
                   'HML3D_vec': motion.squeeze(),
                   'positions': positions.squeeze().detach().cpu().numpy(),
                   'vertuces': vertices,
                   'faces': faces}
        with open(f"{sav_path}/motionllm_{level}_{iterate}_predicted.pk", "wb") as f:
            pickle.dump(my_dict, f)






        # Leader:
        the_other_motion_tokens = torch.cat(the_other_motion_tokens)
        motion = model.net.forward_decoder(the_other_motion_tokens)
        motion = model.denormalize(motion.detach().cpu().numpy())
        positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
        # print(motion.shape)

        iterate = 0
        if True:
            sav_path = f"./demo/eval4pair/{level}"

            os.makedirs(sav_path, exist_ok=True)
            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_input.mp4"),
                           t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                           title=(f"Input {'follower' if current_batch_task=='follower_to_leader' else 'leader'}: {Style} {Pair}"),
                           fps=20, radius=4)

            vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                                              sav_path,
                                              name=f'motionllm_{level}_{iterate}_3DMesh_input.mp4')

            # For Shay
            my_dict = {'caption': "infer",
                       'HML3D_vec': motion.squeeze(),
                       'positions': positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_input.pk", "wb") as f:
                pickle.dump(my_dict, f)

        leader_motion_tokens = item
'''
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval_new"
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
'''

from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
def load_data_pair(style='beginner', pair="Pair1", start_sec=0, end_sec=5):
    args = args = get_args_parser()

    train_dataset = Salsa_Dataset(args,
                    lmdb_dir='utils/salsa_utils/Salsa_Temp/lmdb_Salsa_pair/lmdb_train',
                    n_poses=100,
                    subdivision_stride=50,
                    pose_resampling_fps=20)
    item = train_dataset.__getitem__(1)
    print()

    #   Workaround striding
    #   pick style and pair:
    # style = "beginner"
    # pair = "Pair1"
    # start_sec, end_sec = 20, 35
    # find the indices '-->'
    list_of_indecis = []
    list_of_aux = []

    for index in range(len(train_dataset)):
        level, ms_desc_L, ms_des_F, \
            vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = train_dataset.__getitem__(index)

        if level==style and \
                aux_info['start_time']>=start_sec and \
                aux_info['end_time'] <= end_sec and \
                pair==aux_info['vid'][:5]: #todo
            if len(list_of_aux)>0:
                if aux_info['start_time']>=list_of_aux[-1]['end_time']:
                    list_of_indecis.append(index)
                    list_of_aux.append(aux_info)
            else:
                list_of_indecis.append(index)
                list_of_aux.append(aux_info)

        if len(list_of_indecis)>0 and \
                list_of_aux[-1]['end_time'] == end_sec and \
                list_of_aux[0]['start_time'] == start_sec:
            break



    items = [train_dataset.__getitem__(index) for index in list_of_indecis]
    return items

if __name__ == "__main__":

    # motion_agent_demo()


    # motionllm_evaluation_qualitative()

    motionllm_evaluation_qualitative_pairs()

